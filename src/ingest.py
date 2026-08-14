"""
Le o documento de referencia (PDF ou CSV), divide em pedacos (chunks),
gera embeddings locais e monta um vector store FAISS.

Uso:
    python -m src.ingest data/documento.pdf
"""

import re
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_document(path: str):
    """Carrega o documento de acordo com a extensao (.pdf ou .csv)."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".csv":
        loader = CSVLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"Extensao nao suportada: {ext}. Use .pdf ou .csv")
    return loader.load()


def limpar_texto(texto: str) -> str:
    """Remove ruidos comuns de PDFs (rodapes, numeros de pagina, sumario).

    Os padroes abaixo foram definidos olhando o texto real dos documentos
    da Receita (Manual DASN-SIMEI e Perguntas e Respostas MEI).
    """
    linhas_limpas = []
    for linha in texto.split("\n"):
        l = linha.strip()
        if not l:
            continue                              # linha vazia
        if re.fullmatch(r"Pág\.\s*\d+", l):
            continue                              # marcador "Pág. 12"
        if l == "Voltar ao Sumário":
            continue                              # rodape de navegacao
        if re.fullmatch(r"\d{1,3}", l):
            continue                              # numero de pagina isolado
        if re.search(r"\.{4,}", l):
            continue                              # linha do sumario (pontilhado)
        linhas_limpas.append(l)
    texto = "\n".join(linhas_limpas)
    texto = re.sub(r"[ \t]{2,}", " ", texto)      # colapsa espacos repetidos
    return texto


def build_vectorstore(path: str, index_dir: str = INDEX_DIR):
    """Le o documento, limpa, divide em chunks, gera embeddings e salva o indice."""
    documents = load_document(path)

    # Limpa o texto de cada pagina ANTES de dividir em chunks
    for doc in documents:
        doc.page_content = limpar_texto(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Documento dividido em {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"Indice salvo em '{index_dir}/'.")
    return vectorstore


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python -m src.ingest <caminho_do_documento>")
        sys.exit(1)
    build_vectorstore(sys.argv[1])
