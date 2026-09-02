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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ingest import build_vectorstore, INDEX_DIR, PASTA_DADOS
from src.agent import load_agent, ask, formatar_fontes, sem_resposta, DATA_BASE

PASTA_WEB = Path(__file__).parent / "web"

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


class Consulta(BaseModel):
    pergunta: str = Field(min_length=3, max_length=500)


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
def perguntar(consulta: Consulta):
    chain = estado.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="O agente ainda está carregando.")

    try:
        resultado = ask(chain, consulta.pergunta)
    except Exception as erro:
        print(f"Erro ao consultar o agente: {erro}")
        raise HTTPException(
            status_code=500, detail="Não foi possível consultar os documentos agora."
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
