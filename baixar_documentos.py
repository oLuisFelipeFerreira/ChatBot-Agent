"""
Baixa automaticamente os documentos oficiais que formam a base de
conhecimento do agente, ja organizando cada um na subpasta correta.

Todas as fontes sao oficiais (planalto.gov.br e gov.br). Textos de lei
nao sao protegidos por direito autoral (Lei 9.610/1998, art. 8, IV).

Uso:
    python baixar_documentos.py
"""

import urllib.request
from pathlib import Path

# (subpasta, nome do arquivo, url)
DOCUMENTOS = [
    (
        "legislacao",
        "codigo_civil.html",
        "https://www.planalto.gov.br/ccivil_03/leis/2002/l10406compilada.htm",
    ),
    (
        "legislacao",
        "lc123_simples_nacional.html",
        "https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp123.htm",
    ),
    (
        "registro-empresarial",
        "manual_empresario_individual.pdf",
        "https://www.gov.br/empresas-e-negocios/pt-br/drei/legislacao/arquivos/"
        "legislacoes-federais/anexoiieinovoindice28dez22.pdf",
    ),
    (
        "registro-empresarial",
        "manual_sociedade_limitada.pdf",
        "https://www.gov.br/empresas-e-negocios/pt-br/drei/legislacao/arquivos/"
        "legislacoes-federais/in-81-2020-anexo-iv-manual-de-ltda-alterado-pela-"
        "in-55-de-2021-revisado-10jun2021.pdf",
    ),
    (
        "registro-empresarial",
        "in_drei_81.pdf",
        "https://www.gov.br/empresas-e-negocios/pt-br/drei/legislacao/"
        "instrucoes-normativas/arquivos-instrucoes-normativas-em-vigor/"
        "INDREI81_Atualizada_in0125.pdf",
    ),
]

# Alguns sites do governo recusam requisicoes sem User-Agent de navegador
CABECALHOS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

BASE = Path("data")


def baixar(subpasta: str, nome: str, url: str) -> bool:
    destino = BASE / subpasta / nome
    destino.parent.mkdir(parents=True, exist_ok=True)

    if destino.exists():
        print(f"  [ja existe] {subpasta}/{nome}")
        return True

    try:
        req = urllib.request.Request(url, headers=CABECALHOS)
        with urllib.request.urlopen(req, timeout=90) as resposta:
            conteudo = resposta.read()
        destino.write_bytes(conteudo)
        tamanho = len(conteudo) / 1024
        print(f"  [OK] {subpasta}/{nome}  ({tamanho:.0f} KB)")
        return True
    except Exception as erro:
        print(f"  [FALHOU] {subpasta}/{nome}")
        print(f"           motivo: {erro}")
        print(f"           baixe manualmente em: {url}")
        return False


def main():
    print("\nBaixando documentos oficiais para a base de conhecimento...\n")
    ok = 0
    for subpasta, nome, url in DOCUMENTOS:
        if baixar(subpasta, nome, url):
            ok += 1

    print(f"\n{ok} de {len(DOCUMENTOS)} documento(s) prontos em '{BASE}/'.")

    if ok < len(DOCUMENTOS):
        print("\nAlgum download falhou? Sem problema: baixe pelo navegador e")
        print("salve na subpasta indicada. O agente funciona com o que houver.")

    print("\nProximo passo: apague a pasta faiss_index/ e rode o agente")
    print("para reindexar com a base nova.\n")


if __name__ == "__main__":
    main()
