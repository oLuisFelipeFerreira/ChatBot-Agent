"""
Busca LITERAL dentro dos chunks indexados (tipo um "Ctrl+F" no indice).

Serve para responder a pergunta: "esse texto chegou a entrar no indice?"
Diferente do inspecionar_indice (que testa a BUSCA), aqui olhamos o
CONTEUDO cru -- sem modelo, sem API, instantaneo.

Uso:
    python -m src.ver_chunks "360.000"
    python -m src.ver_chunks "empresa de pequeno porte" --arquivo lc123
"""

import sys

from .ingest import carregar_chunks


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    termo = sys.argv[1]
    filtro_arquivo = None
    if "--arquivo" in sys.argv:
        i = sys.argv.index("--arquivo")
        if i + 1 < len(sys.argv):
            filtro_arquivo = sys.argv[i + 1].lower()

    chunks = carregar_chunks()
    if not chunks:
        print("Nenhum chunk salvo. Rode antes: python -m src.ingest")
        sys.exit(1)

    print(f"Procurando {termo!r} em {len(chunks)} chunks", end="")
    if filtro_arquivo:
        print(f" (apenas arquivos contendo {filtro_arquivo!r})", end="")
    print("...\n")

    encontrados = 0
    for i, chunk in enumerate(chunks):
        arquivo = chunk.metadata.get("arquivo", "?")
        if filtro_arquivo and filtro_arquivo not in arquivo.lower():
            continue
        if termo.lower() not in chunk.page_content.lower():
            continue

        encontrados += 1
        categoria = chunk.metadata.get("categoria", "?")
        pagina = chunk.metadata.get("page")
        local = arquivo + (f" p.{pagina + 1}" if pagina is not None else "")

        # Mostra o trecho ao redor da ocorrencia
        texto = " ".join(chunk.page_content.split())
        pos = texto.lower().find(termo.lower())
        inicio = max(0, pos - 150)
        fim = min(len(texto), pos + 250)
        print(f"[{encontrados}] chunk #{i}  [{categoria}] {local}")
        print(f"    ...{texto[inicio:fim]}...\n")

        if encontrados >= 8:
            print("(mostrando apenas os 8 primeiros)")
            break

    if encontrados == 0:
        print("NAO ENCONTRADO.")
        print("Isso significa que esse texto nao entrou no indice -- o problema")
        print("esta na LEITURA do documento, nao na busca.")
    else:
        print(f"Total exibido: {encontrados}")
        print("O texto ESTA no indice. Se o agente nao o encontra, o problema")
        print("esta na BUSCA (embeddings/BM25), nao na leitura.")


if __name__ == "__main__":
    main()
