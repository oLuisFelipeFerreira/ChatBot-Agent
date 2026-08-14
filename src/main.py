"""
Ponto de entrada do agente. Roda a indexacao (se ainda nao existir) e
abre um loop de perguntas e respostas no terminal.

Uso:
    python -m src.main data/documento.pdf
"""

import sys
from pathlib import Path

from .ingest import build_vectorstore, INDEX_DIR
from .agent import load_agent, ask


def main():
    # Sem argumento, usa a pasta data/ inteira.
    doc_path = sys.argv[1] if len(sys.argv) > 1 else "data"

    if not Path(INDEX_DIR).exists():
        print("Indice nao encontrado, gerando pela primeira vez...")
        build_vectorstore(doc_path)

    chain = load_agent()
    print("\nAgente pronto. Digite sua pergunta (ou 'sair' para encerrar).\n")

    while True:
        pergunta = input("Voce: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue

        resultado = ask(chain, pergunta)
        print(f"\nAgente: {resultado['result']}")

        # Cita a fonte: nome do arquivo e pagina de cada trecho usado
        fontes = set()
        for doc in resultado["source_documents"]:
            nome = Path(doc.metadata.get("source", "?")).name
            pagina = doc.metadata.get("page")
            if pagina is not None:
                fontes.add(f"{nome} (pag. {pagina + 1})")
            else:
                fontes.add(nome)
        if fontes:
            print(f"Fontes: {', '.join(sorted(fontes))}\n")
        else:
            print()


if __name__ == "__main__":
    main()
