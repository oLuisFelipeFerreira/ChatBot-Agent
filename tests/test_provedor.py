"""
Teste offline da troca de provedor (nao faz chamada de API, nao gasta nada).

Garante que criar_llm() faz as tres coisas certas:
  1. monta o modelo do provedor configurado;
  2. avisa com clareza quando falta a chave;
  3. avisa com clareza quando falta o pacote do provedor.

Uso:
    python test_provedor.py
"""

import os

from src import agent


def com_config(provedor, modelo):
    """Troca o provedor so dentro do teste."""
    agent.PROVEDOR, agent.MODELO = provedor, modelo


def test_padrao_e_local_e_sem_chave():
    """O padrao do projeto roda na maquina de quem clonou, sem chave.

    Se alguem trocar isto por um provedor de API, este teste quebra -- e
    deve quebrar: seria mudar a promessa de "clone e teste de graca".
    """
    assert agent.PROVEDOR == "ollama"
    assert agent.CHAVE_DO_PROVEDOR["ollama"] is None


def test_monta_o_modelo_do_provedor_escolhido():
    com_config("anthropic", "claude-haiku-4-5-20251001")
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "chave-de-teste")
    assert type(agent.criar_llm()).__name__ == "ChatAnthropic"


def test_chave_faltando_avisa_qual():
    com_config("groq", "llama-3.3-70b-versatile")
    os.environ.pop("GROQ_API_KEY", None)
    try:
        agent.criar_llm()
    except EnvironmentError as erro:
        assert "GROQ_API_KEY" in str(erro)
    else:
        raise AssertionError("deveria ter reclamado da chave faltando")


def test_pacote_faltando_diz_o_que_instalar():
    com_config("groq", "llama-3.3-70b-versatile")
    os.environ["GROQ_API_KEY"] = "chave-de-teste"
    try:
        agent.criar_llm()
    except ImportError as erro:
        assert "langchain-groq" in str(erro)
    else:
        print("  (langchain-groq instalado -- nada a testar aqui)")


if __name__ == "__main__":
    original = (agent.PROVEDOR, agent.MODELO)
    for nome, funcao in list(globals().items()):
        if nome.startswith("test_"):
            funcao()
            print(f"OK  {nome}")
    com_config(*original)
    print("\nTudo certo.")
