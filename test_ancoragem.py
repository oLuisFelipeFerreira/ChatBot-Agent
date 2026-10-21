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


def test_resposta_normal_nao_e_recusa():
    assert not sem_resposta("O MEI paga um valor fixo mensal por meio do DAS.")


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


if __name__ == "__main__":
    for nome, funcao in list(globals().items()):
        if nome.startswith("test_"):
            funcao()
            print(f"OK  {nome}")
    print("\nTudo certo.")
