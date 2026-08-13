"""
Monta a chain de recuperacao + resposta (RAG) usando Claude (Anthropic)
como modelo de linguagem e o indice FAISS gerado pelo ingest.py.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from .ingest import INDEX_DIR, EMBEDDING_MODEL

load_dotenv()


def load_agent(index_dir: str = INDEX_DIR):
    """Carrega o indice salvo e monta a chain de perguntas e respostas."""
    if not Path(index_dir).exists():
        raise FileNotFoundError(
            f"Indice '{index_dir}' nao encontrado. Rode antes: "
            f"python -m src.ingest data/seu_documento.pdf"
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY nao encontrada. Copie .env.example para .env "
            "e preencha com sua chave."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # claude-sonnet-5: bom equilibrio de qualidade e custo para RAG.
    # Se quiser economizar chamadas de API durante os testes, troque por
    # "claude-haiku-4-5-20251001" (mais rapido e mais barato).
    llm = ChatAnthropic(model="claude-sonnet-5")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return chain


def ask(chain, pergunta: str) -> str:
    """Faz uma pergunta ao agente e devolve a resposta em texto."""
    resultado = chain.invoke({"query": pergunta})
    return resultado["result"]
