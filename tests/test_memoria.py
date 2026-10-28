"""
Teste do historico de conversa. Nao usa rede nem modelo: so verifica a
traducao do historico que o cliente manda para as mensagens do LangChain.

Uso:
    python test_memoria.py
"""

from src.agent import montar_historico, MAX_HISTORICO


def test_sem_historico_devolve_lista_vazia():
    assert montar_historico(None) == []
    assert montar_historico([]) == []


def test_alterna_pergunta_e_resposta_na_ordem():
    m = montar_historico([
        {"pergunta": "p1", "resposta": "r1"},
        {"pergunta": "p2", "resposta": "r2"},
    ])
    assert [type(x).__name__ for x in m] == [
        "HumanMessage", "AIMessage", "HumanMessage", "AIMessage"
    ]
    assert [x.content for x in m] == ["p1", "r1", "p2", "r2"]


def test_mantem_so_as_ultimas_trocas():
    """O fim da conversa e o que da sentido a pergunta atual."""
    trocas = [{"pergunta": f"p{i}", "resposta": f"r{i}"} for i in range(10)]
    m = montar_historico(trocas)
    assert len(m) == MAX_HISTORICO * 2
    assert m[0].content == "p6"   # as seis primeiras trocas foram cortadas
    assert m[-1].content == "r9"


def test_ignora_campos_vazios_ou_ausentes():
    """O cliente manda o historico; ele pode vir torto."""
    m = montar_historico([
        {"pergunta": "   ", "resposta": "r1"},
        {"resposta": "so resposta"},
        {"pergunta": "so pergunta"},
    ])
    assert [x.content for x in m] == ["r1", "so resposta", "so pergunta"]


if __name__ == "__main__":
    for nome, funcao in list(globals().items()):
        if nome.startswith("test_"):
            funcao()
            print(f"OK  {nome}")
    print("\nTudo certo.")
