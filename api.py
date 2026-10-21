"""
API do Assistente Empresarial.

Separa o "cerebro" (o agente RAG, em src/) da "cara" (o front em web/).
O FastAPI faz duas coisas:
  1. expoe o endpoint POST /api/perguntar, que devolve resposta + fontes
  2. serve os arquivos estaticos do front

Assim tudo roda em UMA porta, sem problema de CORS e com um comando so.

Uso:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ingest import build_vectorstore, INDEX_DIR, PASTA_DADOS
from src.agent import (
    load_agent, ask, formatar_fontes, sem_resposta, truncar_na_recusa,
    montar_historico, DATA_BASE,
)
from src.limite import Limite
from src.registro import registrar
from src.agent import PROVEDOR, MODELO

PASTA_WEB = Path(__file__).parent / "web"

# Quanto cada visitante pode perguntar, e quanto o site todo gasta por dia.
# Ajustaveis pelo .env conforme a cota do provedor que voce estiver usando.
limite = Limite(
    por_ip_minuto=int(os.getenv("LIMITE_POR_MINUTO", "6")),
    global_dia=int(os.getenv("LIMITE_POR_DIA", "300")),
)

# Atras de um proxy (nginx, Cloudflare), o IP real vem no X-Forwarded-For e
# request.client.host seria sempre o do proxy -- todo mundo contaria como um
# visitante so. Fora de um proxy vale o contrario: o cabecalho e escrito pelo
# cliente, entao confiar nele deixaria qualquer um furar o limite inventando
# um IP. Por isso so lemos o cabecalho quando voce avisa que ha proxy.
ATRAS_DE_PROXY = os.getenv("ATRAS_DE_PROXY", "").lower() in {"1", "true", "sim"}


def identificar(request: Request) -> str:
    if ATRAS_DE_PROXY:
        encaminhado = request.headers.get("x-forwarded-for", "")
        if encaminhado:
            return encaminhado.split(",")[0].strip()
    return request.client.host if request.client else "desconhecido"

# Guarda o agente carregado. Fica em memoria entre as requisicoes -- carregar
# o modelo a cada pergunta seria lento demais.
estado = {"chain": None}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Roda uma vez, quando o servidor sobe."""
    if not Path(INDEX_DIR).exists():
        print("Indice nao encontrado. Gerando pela primeira vez...")
        build_vectorstore(PASTA_DADOS)
    print("Carregando o agente...")
    estado["chain"] = load_agent()
    print("Agente pronto.")
    yield
    estado["chain"] = None


app = FastAPI(title="Assistente Empresarial", lifespan=ciclo_de_vida)


class Troca(BaseModel):
    """Uma pergunta ja feita e a resposta que ela recebeu."""
    pergunta: str = Field(max_length=500)
    resposta: str = Field(max_length=5000)


class Consulta(BaseModel):
    pergunta: str = Field(min_length=3, max_length=500)

    # O historico vem do CLIENTE, nao de uma sessao no servidor. A API
    # continua sem estado: nada para limpar, nada que se perca no restart,
    # e funciona igual com varios processos.
    #
    # O preco de confiar no cliente e que ele pode mandar qualquer coisa --
    # inclusive um historico enorme, que encareceria a chamada. Por isso o
    # teto aqui: max_length corta o numero de trocas, e os limites de Troca
    # cortam o tamanho de cada uma.
    historico: list[Troca] = Field(default_factory=list, max_length=10)


class Trecho(BaseModel):
    """O texto exato que embasou a resposta, para o usuario conferir."""
    pagina: int | None = None
    texto: str


class Fonte(BaseModel):
    categoria: str
    arquivo: str
    paginas: list[int]
    # Caminho relativo a data/, para o front montar o link ate o documento.
    caminho: str = ""
    # Sem declarar aqui, o response_model do FastAPI descartaria os trechos
    # em silencio -- sem erro, sem aviso, so sumiriam da resposta.
    trechos: list[Trecho] = Field(default_factory=list)


class Resposta(BaseModel):
    resposta: str
    fontes: list[Fonte]


@app.get("/api/info")
def info():
    """Dados que o front usa para se apresentar."""
    return {"base_atualizada_em": DATA_BASE}


@app.post("/api/perguntar", response_model=Resposta)
def perguntar(consulta: Consulta, request: Request):
    recusa = limite.checar(identificar(request))
    if recusa:
        raise HTTPException(status_code=429, detail=recusa)

    chain = estado.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="O agente ainda está carregando.")

    comecou = time.monotonic()
    try:
        resultado = ask(
            chain,
            consulta.pergunta,
            historico=[t.model_dump() for t in consulta.historico],
        )
    except Exception as erro:
        print(f"Erro ao consultar o agente: {erro}")
        registrar(
            pergunta=consulta.pergunta,
            erro=type(erro).__name__,
            ms=int((time.monotonic() - comecou) * 1000),
            provedor=PROVEDOR,
            modelo=MODELO,
        )
        texto_erro = str(erro).lower()

        # COTA DIARIA do provedor esgotada (tipico de camada gratuita).
        # Diferente de congestionamento: nao adianta tentar de novo daqui a
        # pouco, so volta amanha. Por isso 429 e nao 503 -- o front NAO
        # retenta 429, e tres tentativas inuteis contra uma cota zerada so
        # fariam o usuario esperar mais para ler a mesma coisa.
        if "quota" in texto_erro or "exceeded your current quota" in texto_erro:
            raise HTTPException(
                status_code=429,
                detail=(
                    "A cota gratuita de consultas deste site acabou por hoje. "
                    "Tente novamente amanhã."
                ),
            )

        # 529 (overloaded) e 429 por excesso de velocidade sao passageiros:
        # o servico esta congestionado. Aqui vale retentar, entao 503.
        if "overloaded" in texto_erro or "529" in texto_erro or "rate limit" in texto_erro:
            raise HTTPException(
                status_code=503,
                detail="O serviço está congestionado no momento.",
            )

        # 401/403: chave invalida, vencida ou sem credito. Erro de
        # configuracao, nao de uso -- vale dizer isso, senao quem instalou o
        # projeto fica caçando um problema que nao existe no codigo.
        if "401" in texto_erro or "403" in texto_erro or "authentication" in texto_erro:
            raise HTTPException(
                status_code=500,
                detail="Chave de API inválida ou sem crédito. Confira o .env.",
            )
        raise HTTPException(
            status_code=500,
            detail="Não foi possível consultar os documentos agora.",
        )

    # Corta qualquer coisa que venha depois da recusa: ja vimos um modelo
    # recusar certo e inventar um imposto no paragrafo seguinte.
    texto = truncar_na_recusa(resultado["result"])
    # Sem resposta encontrada = sem fontes. Listar documentos que nao
    # embasaram nada passaria a impressao errada.
    nao_achou = sem_resposta(texto)
    fontes = [] if nao_achou else formatar_fontes(resultado["source_documents"])

    registrar(
        pergunta=consulta.pergunta,
        achou=not nao_achou,
        continuacao=bool(consulta.historico),
        n_fontes=len(fontes),
        ms=int((time.monotonic() - comecou) * 1000),
        provedor=PROVEDOR,
        modelo=MODELO,
    )
    return {"resposta": texto, "fontes": fontes}


def _evento(tipo: str, dado) -> bytes:
    """Uma linha do fluxo: um objeto JSON seguido de quebra de linha.

    O formato e NDJSON: {"tipo": "texto", "dado": "..."} por linha. O
    cliente separa por quebra de linha e decodifica cada pedaco.
    """
    linha = json.dumps({"tipo": tipo, "dado": dado}, ensure_ascii=False)
    return (linha + "\n").encode()


@app.post("/api/perguntar/stream")
async def perguntar_stream(consulta: Consulta, request: Request):
    """Mesma pergunta, resposta em pedacos conforme o modelo escreve.

    Medido: a resposta completa leva de 10 a 18 segundos. Esperar tudo isso
    olhando um "carregando" faz o site parecer travado; com o texto
    aparecendo, a espera continua a mesma e a sensacao muda por completo.

    O formato e NDJSON (uma linha JSON por evento), e nao SSE, porque o
    EventSource do navegador so faz GET -- e a pergunta e o historico
    precisam ir no corpo de um POST.
    """
    recusa = limite.checar(identificar(request))
    if recusa:
        raise HTTPException(status_code=429, detail=recusa)

    chain = estado.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="O agente ainda está carregando.")

    async def gerar():
        comecou = time.monotonic()
        documentos, texto = [], ""
        try:
            async for pedaco in chain.astream({
                "input": consulta.pergunta,
                "chat_history": montar_historico(
                    [t.model_dump() for t in consulta.historico]
                ),
            }):
                # A busca termina antes da escrita, entao o 'context' chega
                # primeiro. Guardamos e so mandamos no fim: se a resposta
                # for "nao encontrei", nenhuma fonte deve aparecer.
                if pedaco.get("context"):
                    documentos = pedaco["context"]
                if pedaco.get("answer"):
                    texto += pedaco["answer"]
                    yield _evento("texto", pedaco["answer"])

            # Mesmo corte do endpoint sincrono. No streaming o texto ja
            # foi transmitido, entao avisamos o front para substituir o
            # que ele mostrou pela recusa limpa.
            limpo = truncar_na_recusa(texto)
            if limpo != texto:
                yield _evento("substituir", limpo)
            fontes = [] if sem_resposta(texto) else formatar_fontes(documentos)
            yield _evento("fontes", fontes)
            registrar(
                pergunta=consulta.pergunta,
                achou=not sem_resposta(texto),
                continuacao=bool(consulta.historico),
                n_fontes=len(fontes),
                ms=int((time.monotonic() - comecou) * 1000),
                provedor=PROVEDOR, modelo=MODELO, streaming=True,
            )
        except Exception as erro:
            # Aqui o status HTTP 200 ja foi enviado com o primeiro byte e
            # nao pode mais virar 500. Por isso o erro viaja como EVENTO
            # dentro do fluxo, e o front o trata como trataria um erro HTTP.
            print(f"Erro ao transmitir a resposta: {erro}")
            registrar(
                pergunta=consulta.pergunta, erro=type(erro).__name__,
                ms=int((time.monotonic() - comecou) * 1000),
                provedor=PROVEDOR, modelo=MODELO, streaming=True,
            )
            texto_erro = str(erro).lower()
            if "quota" in texto_erro:
                aviso = ("A cota gratuita de consultas deste site acabou por hoje. "
                         "Tente novamente amanhã.")
            elif "overloaded" in texto_erro or "529" in texto_erro or "rate limit" in texto_erro:
                aviso = "O serviço está congestionado agora. Tente de novo em instantes."
            else:
                aviso = "Não foi possível consultar os documentos agora."
            yield _evento("erro", aviso)

    return StreamingResponse(gerar(), media_type="application/x-ndjson")


# --- Documentos originais ----------------------------------------------
# Servir data/ e o que torna a fonte VERIFICAVEL: o front monta um link
# para o arquivo na pagina citada, e o usuario confere a resposta contra o
# documento oficial. Sem isto, "pagina 39" e texto morto.
#
# Expor a pasta inteira e aceitavel aqui porque sao documentos publicos do
# governo -- e justamente o que o projeto promete mostrar.
if Path(PASTA_DADOS).exists():
    app.mount("/documentos", StaticFiles(directory=PASTA_DADOS), name="documentos")


# --- Front-end estatico -------------------------------------------------
# Montado por ultimo para nao capturar as rotas /api/*
if PASTA_WEB.exists():
    app.mount("/", StaticFiles(directory=PASTA_WEB, html=True), name="web")
else:
    @app.get("/")
    def sem_front():
        return {"aviso": "Pasta web/ nao encontrada. A API continua em /api/perguntar."}
