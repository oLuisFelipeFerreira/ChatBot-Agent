"""
Ponto de entrada do agente no terminal. Roda a indexacao (se ainda nao
existir) e abre um loop de perguntas e respostas.

Uso:
    python -m src.main            # usa a pasta data/ inteira
    python -m src.main data/legislacao
"""

import sys
from pathlib import Path

from .ingest import build_vectorstore, INDEX_DIR, PASTA_DADOS
from .agent import load_agent, ask, formatar_fontes, DATA_BASE


def main():
    doc_path = sys.argv[1] if len(sys.argv) > 1 else PASTA_DADOS

    if not Path(INDEX_DIR).exists():
        print("Indice nao encontrado, gerando pela primeira vez...")
        build_vectorstore(doc_path)

    chain = load_agent()
    print(f"\nAssistente de Direito Empresarial (base: {DATA_BASE})")
    print("Digite sua pergunta (ou 'sair' para encerrar).\n")

    while True:
        pergunta = input("Voce: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue

        resultado = ask(chain, pergunta)
        print(f"\nAgente: {resultado['result']}\n")

        fontes = formatar_fontes(resultado["source_documents"])
        if fontes:
            print("Fontes consultadas:")
            for f in fontes:
                paginas = f" (pag. {', '.join(map(str, f['paginas']))})" if f["paginas"] else ""
                print(f"  [{f['categoria']}] {f['arquivo']}{paginas}")
            print()


if __name__ == "__main__":
    main()
