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
    if len(sys.argv) != 2:
        print("Uso: python -m src.main <caminho_do_documento>")
        sys.exit(1)

    doc_path = sys.argv[1]

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
        resposta = ask(chain, pergunta)
        print(f"\nAgente: {resposta}\n")


if __name__ == "__main__":
    main()
