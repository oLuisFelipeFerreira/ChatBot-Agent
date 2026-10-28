"""
Teste da ancoragem: o que o agente faz quando NAO sabe.

Esta e a regra mais importante do projeto. Um assistente juridico que
inventa e pior do que nenhum, entao a recusa precisa ser confiavel --
inclusive quando o modelo recusa certo e continua falando errado.

Uso:
    python test_ancoragem.py
"""

from src.agent import sem_resposta, truncar_na_recusa, FRASE_SEM_RESPOSTA


def test_reconhece_a_recusa():
    assert sem_resposta("Não encontrei essa informação nos documentos disponíveis.")
    assert sem_resposta("NÃO ENCONTREI ESSA INFORMAÇÃO NOS DOCUMENTOS DISPONÍVEIS")


def test_reconhece_a_recusa_parafraseada():
    """Casos REAIS da avaliacao que a frase exata nao pegava.

    Quando a recusa passa despercebida, o app mostra a lista de fontes
    junto de um "nao sei" -- o oposto do que essa logica existe para
    fazer. O modelo parafraseia; o casamento nao pode ser literal.
    """
    assert sem_resposta(
        "Não encontrei informações nos documentos disponíveis sobre "
        "abertura de empresa em Portugal. Posso ajudar com dúvidas "
        "sobre Direito Empresarial brasileiro?"
    )
    assert sem_resposta(
        "Não encontrei a informação necessária nos documentos "
        "disponíveis para responder a essa pergunta."
    )
    assert sem_resposta("**Não encontrei essa informação nos documentos disponíveis.**")


def test_resposta_normal_nao_e_recusa():
    assert not sem_resposta("O MEI paga um valor fixo mensal por meio do DAS.")


def test_resposta_com_ressalva_nao_e_recusa():
    """Responder e ressalvar o que falta NAO e recusar.

    Tratar isto como recusa esconderia as fontes de uma resposta boa.
    """
    assert not sem_resposta(
        "O limite é R$ 81.000 por ano. Não encontrei o valor de anos "
        "anteriores nos documentos."
    )
    assert not sem_resposta(
        "Não encontrei problemas na documentação apresentada à junta comercial."
    )


def test_corta_alucinacao_depois_da_recusa():
    """Caso real: um modelo 3B recusou certo e inventou um tributo."""
    alucinado = (
        "Não encontrei essa informação nos documentos disponíveis.\n\n"
        "O Imposto Municipal Sobre a Renda (IMR) é calculado com base na "
        "receita bruta da empresa."
    )
    limpo = truncar_na_recusa(alucinado)
    assert limpo == FRASE_SEM_RESPOSTA + "."
    assert "IMR" not in limpo
    assert "Imposto Municipal" not in limpo


def test_nao_mexe_em_resposta_valida():
    boa = "O MEI paga 5% do salário-mínimo, conforme o art. 18-A da LC 123."
    assert truncar_na_recusa(boa) == boa


def test_nao_corta_continuacao_no_mesmo_paragrafo():
    """Limite conhecido, registrado de proposito.

    O corte e por paragrafo. Uma continuacao na MESMA linha passa. Este
    teste existe para que o limite seja uma decisao consciente, e nao uma
    surpresa -- se um dia a regra mudar, ele quebra e obriga a pensar.
    """
    mesmo_paragrafo = (
        "Não encontrei essa informação nos documentos disponíveis. "
        "A pergunta parece se referir a um caso específico."
    )
    assert truncar_na_recusa(mesmo_paragrafo) == mesmo_paragrafo


if __name__ == "__main__":
    for nome, funcao in list(globals().items()):
        if nome.startswith("test_"):
            funcao()
            print(f"OK  {nome}")
    print("\nTudo certo.")
