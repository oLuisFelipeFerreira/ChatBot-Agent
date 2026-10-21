"""
Roda o conjunto de avaliacao e mede o sistema -- nao o modelo.

Por que isto existe: sem medir, "melhorou" e opiniao. Ajustar TOP_K, o
peso da busca hibrida ou o modelo vira adivinhacao, porque nada diz se o
resultado ficou melhor ou pior. Com 15 perguntas anotadas, cada mudanca
vira um numero comparavel.

O que ele mede, em ordem de importancia:

  RECUSA     o sistema admite quando nao sabe? E a regra mais importante
             do projeto: um assistente juridico que inventa e pior que
             nenhum.
  ARMADILHA  o sistema evita os valores vencidos que estao no texto
             literal da lei? (ex.: o R$ 45,65 do art. 18-A)
  FONTE      a busca trouxe o documento que de fato contem a resposta?
             Separa falha de BUSCA de falha de MODELO -- se a fonte certa
             veio e a resposta saiu errada, o problema e o modelo.
  CONTEUDO   a resposta contem os numeros e termos esperados?

Uso:
    python -m avaliacao.rodar                      # usa o .env
    LLM_PROVIDER=google_genai LLM_MODEL=gemini-3.8-flash python -m avaliacao.rodar
    python -m avaliacao.rodar --tabela             # so reimprime a tabela
"""

import json
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

PASTA = Path(__file__).parent
PERGUNTAS = PASTA / "perguntas.json"
RESULTADOS = PASTA / "resultados"


def normalizar(texto: str) -> str:
    """Tira acento e caixa para comparar.

    O modelo escreve "nao" ou "nao"; o gabarito nao pode depender disso.
    """
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.lower()


def avaliar(caso: dict, texto: str, fontes: list) -> dict:
    """Confere uma resposta contra o gabarito. Devolve o que passou e o que nao."""
    from src.agent import sem_resposta

    t = normalizar(texto)
    arquivos = {f["arquivo"] for f in fontes}
    recusou = sem_resposta(texto)
    checagens = {}

    if caso["espera"] == "recusa":
        checagens["recusa"] = recusou
    else:
        checagens["recusa"] = not recusou  # responder quando DEVE responder

        esperadas = caso.get("fontes_esperadas") or []
        if esperadas:
            checagens["fonte"] = any(a in arquivos for a in esperadas)

        if caso.get("deve_conter"):
            checagens["conteudo"] = all(
                normalizar(x) in t for x in caso["deve_conter"]
            )
        if caso.get("deve_conter_algum"):
            checagens["conteudo"] = checagens.get("conteudo", True) and any(
                normalizar(x) in t for x in caso["deve_conter_algum"]
            )

    if caso.get("nao_pode_conter"):
        checagens["armadilha"] = not any(
            normalizar(x) in t for x in caso["nao_pode_conter"]
        )

    return checagens


def rodar():
    from src.agent import load_agent, ask, formatar_fontes, truncar_na_recusa
    from src.agent import PROVEDOR, MODELO, TOP_K, USAR_MULTIQUERY

    casos = json.loads(PERGUNTAS.read_text(encoding="utf-8"))
    print(f"\nModelo: {PROVEDOR} / {MODELO}")
    print(f"TOP_K={TOP_K}  multiquery={USAR_MULTIQUERY}")
    print(f"{len(casos)} perguntas. Carregando o agente...\n")

    chain = load_agent()
    linhas = []

    for i, caso in enumerate(casos, 1):
        t0 = time.monotonic()
        try:
            bruto = ask(chain, caso["pergunta"])
            texto = truncar_na_recusa(bruto["result"])
            fontes = formatar_fontes(bruto["source_documents"])
            erro = None
        except Exception as e:
            texto, fontes, erro = "", [], f"{type(e).__name__}: {e}"
        segundos = time.monotonic() - t0

        checagens = avaliar(caso, texto, fontes) if not erro else {}
        passou = bool(checagens) and all(checagens.values())
        falhou_em = [k for k, v in checagens.items() if not v]

        marca = "OK  " if passou else "FALHA"
        detalhe = f" ({', '.join(falhou_em)})" if falhou_em else ""
        print(f"{marca} [{i:2}/{len(casos)}] {segundos:5.1f}s  {caso['id']}{detalhe}")
        if erro:
            print(f"      erro: {erro}")

        linhas.append({
            "id": caso["id"],
            "pergunta": caso["pergunta"],
            "espera": caso["espera"],
            "passou": passou,
            "checagens": checagens,
            "falhou_em": falhou_em,
            "segundos": round(segundos, 1),
            "fontes": sorted({f["arquivo"] for f in fontes}),
            "resposta": texto,
            "erro": erro,
        })

    resumo = montar_resumo(linhas, PROVEDOR, MODELO, TOP_K, USAR_MULTIQUERY)
    RESULTADOS.mkdir(exist_ok=True)
    # O nome inclui a CONFIGURACAO, nao so o modelo: rodar com TOP_K
    # diferente e outro experimento, e sobrescrever o anterior apagaria a
    # comparacao -- foi o que aconteceu na primeira vez.
    limpo = MODELO.replace(":", "-").replace("/", "-")
    nome = f"{limpo}_k{TOP_K}{'_mq' if USAR_MULTIQUERY else ''}.json"
    destino = RESULTADOS / nome
    destino.write_text(
        json.dumps({"resumo": resumo, "casos": linhas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultado salvo em {destino}")
    imprimir_tabela()


def montar_resumo(linhas, provedor, modelo, top_k, multiquery) -> dict:
    def taxa(filtro, chave):
        casos = [l for l in linhas if filtro(l) and chave in l["checagens"]]
        if not casos:
            return None
        return sum(1 for l in casos if l["checagens"][chave]) / len(casos)

    return {
        "quando": datetime.now().isoformat(timespec="seconds"),
        "provedor": provedor,
        "modelo": modelo,
        "top_k": top_k,
        "multiquery": multiquery,
        "total": len(linhas),
        "passou": sum(1 for l in linhas if l["passou"]),
        "recusa_correta": taxa(lambda l: l["espera"] == "recusa", "recusa"),
        "fonte_correta": taxa(lambda l: True, "fonte"),
        "conteudo_correto": taxa(lambda l: True, "conteudo"),
        "armadilha_evitada": taxa(lambda l: True, "armadilha"),
        "segundos_mediana": sorted(l["segundos"] for l in linhas)[len(linhas) // 2],
    }


def pct(v):
    return "-" if v is None else f"{v*100:.0f}%"


def imprimir_tabela():
    """Imprime a tabela em Markdown, pronta para colar no README."""
    arquivos = sorted(RESULTADOS.glob("*.json")) if RESULTADOS.exists() else []
    if not arquivos:
        print("Nenhum resultado ainda.")
        return

    print("\n--- cole no README -------------------------------------\n")
    print("| Modelo | TOP_K | Acertos | Recusou certo | Fonte certa | Armadilhas | Mediana |")
    print("|---|---|---|---|---|---|---|")
    for a in arquivos:
        r = json.loads(a.read_text(encoding="utf-8"))["resumo"]
        print(
            f"| `{r['modelo']}` | {r['top_k']} | {r['passou']}/{r['total']} | "
            f"{pct(r['recusa_correta'])} | {pct(r['fonte_correta'])} | "
            f"{pct(r['armadilha_evitada'])} | {r['segundos_mediana']}s |"
        )
    print()


if __name__ == "__main__":
    if "--tabela" in sys.argv:
        imprimir_tabela()
    else:
        rodar()
