"""
Ferramenta de DIAGNOSTICO da recuperacao (o "R" do RAG).

Mostra quais trechos cada buscador traz para uma pergunta -- SEM chamar o
Claude, ou seja, sem gastar API. Compara os tres modos:

  [SEMANTICO] busca por significado (embeddings)
  [BM25]      busca por palavra-chave
  [HIBRIDO]   os dois combinados (o que o agente usa de verdade)

Use isto sempre que o agente responder "nao encontrei" para algo que voce
sabe que esta nos documentos.

Uso:
    python -m src.inspecionar_indice
    python -m src.inspecionar_indice "sua pergunta aqui"
"""

import sys
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .ingest import INDEX_DIR, EMBEDDING_MODEL, carregar_chunks
from .busca import montar_retriever_hibrido, tokenizar_pt
from .agent import TOP_K

PERGUNTAS_PADRAO = [
    "Qual a diferença entre MEI, ME e EPP?",
    "O que define juridicamente um empresário?",
    "Qual o limite de faturamento da microempresa?",
]


def mostrar(titulo, documentos, limite=4):
    print(f"\n  --- {titulo} ---")
    if not documentos:
        print("      (nada encontrado)")
        return
    for i, doc in enumerate(documentos[:limite], 1):
        categoria = doc.metadata.get("categoria", "?")
        arquivo = doc.metadata.get("arquivo", "?")
        pagina = doc.metadata.get("page")
        local = arquivo + (f" p.{pagina + 1}" if pagina is not None else "")
        trecho = " ".join(doc.page_content.split())[:150]
        print(f"   [{i}] [{categoria}] {local}")
        print(f"       {trecho}...")


def main():
    if not Path(INDEX_DIR).exists():
        print(f"Indice '{INDEX_DIR}' nao encontrado. Rode antes: python -m src.ingest")
        sys.exit(1)

    print(f"Carregando modelo: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    chunks = carregar_chunks()

    print(f"\nVetores no indice  : {vectorstore.index.ntotal}")
    print(f"Chunks salvos      : {len(chunks)}")
    print(f"Trechos por busca  : {TOP_K}")

    if not chunks:
        print("\nAVISO: sem chunks.json -- reindexe para testar a busca hibrida.")

    semantico = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    hibrido = montar_retriever_hibrido(vectorstore, chunks, k=TOP_K) if chunks else None

    perguntas = sys.argv[1:] if len(sys.argv) > 1 else PERGUNTAS_PADRAO

    for pergunta in perguntas:
        print("\n" + "=" * 74)
        print(f"PERGUNTA: {pergunta}")
        print(f"tokens BM25: {tokenizar_pt(pergunta)}")
        print("=" * 74)

        mostrar("SEMANTICO (embeddings)", semantico.invoke(pergunta))
        if hibrido:
            mostrar("HIBRIDO (BM25 + semantico)  <- o que o agente usa",
                    hibrido.invoke(pergunta))


if __name__ == "__main__":
    main()
