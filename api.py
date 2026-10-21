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

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ingest import build_vectorstore, INDEX_DIR, PASTA_DADOS
from src.agent import load_agent, ask, formatar_fontes, sem_resposta, DATA_BASE
from src.limite import Limite

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


class Fonte(BaseModel):
    categoria: str
    arquivo: str
    paginas: list[int]


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

    try:
        resultado = ask(
            chain,
            consulta.pergunta,
            historico=[t.model_dump() for t in consulta.historico],
        )
    except Exception as erro:
        print(f"Erro ao consultar o agente: {erro}")
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

    texto = resultado["result"]
    # Sem resposta encontrada = sem fontes. Listar documentos que nao
    # embasaram nada passaria a impressao errada.
    fontes = [] if sem_resposta(texto) else formatar_fontes(resultado["source_documents"])
    return {"resposta": texto, "fontes": fontes}


# --- Front-end estatico -------------------------------------------------
# Montado por ultimo para nao capturar as rotas /api/*
if PASTA_WEB.exists():
    app.mount("/", StaticFiles(directory=PASTA_WEB, html=True), name="web")
else:
    @app.get("/")
    def sem_front():
        return {"aviso": "Pasta web/ nao encontrada. A API continua em /api/perguntar."}
