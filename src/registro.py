"""
Registro das consultas, uma linha JSON por pergunta.

Para que serve, em ordem de utilidade:

1. SABER O QUE PERGUNTAM. A lista de perguntas reais e a unica fonte
   honesta de prioridade: ela diz o que a base precisa cobrir, em vez de
   voce adivinhar.
2. ACHAR AS LACUNAS. As perguntas que receberam "nao encontrei" apontam
   exatamente onde falta documento -- ou onde a busca esta falhando.
3. MEDIR A LENTIDAO. Sem numero, "esta lento" e opiniao. Com a latencia
   registrada, da para decidir se vale otimizar e provar que melhorou.
4. ALIMENTAR A AVALIACAO. Perguntas reais sao material melhor para o
   conjunto de testes do que perguntas inventadas por quem fez o sistema.

O que NAO e registrado, de proposito:
  - o IP de quem perguntou: nao e necessario para nenhum dos usos acima,
    e dado pessoal que voce nao precisa guardar e melhor nao guardar.
  - o texto completo da resposta: encheria o arquivo e nao acrescenta
    nada que o resto da linha ja nao diga.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ARQUIVO = Path(os.getenv("ARQUIVO_REGISTRO", "logs/consultas.jsonl"))


def registrar(**campos) -> None:
    """Acrescenta uma linha ao registro.

    NUNCA levanta excecao: disco cheio, pasta sem permissao ou caminho
    invalido nao podem derrubar uma pergunta que ja foi respondida.
    Observabilidade que quebra o sistema observado e pior que nenhuma.
    """
    try:
        ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
        linha = {"quando": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        linha.update(campos)
        with ARQUIVO.open("a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
    except Exception as erro:
        print(f"[registro] nao consegui gravar: {erro}")
