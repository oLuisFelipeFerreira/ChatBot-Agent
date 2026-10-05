"""
Teste do limite de requisicoes. Nao usa rede nem espera de verdade:
o tempo e injetado, entao roda instantaneo.

Uso:
    python test_limite.py
"""

from src.limite import Limite


def test_deixa_passar_dentro_do_limite():
    limite = Limite(por_ip_minuto=3)
    for _ in range(3):
        assert limite.checar("1.1.1.1", agora=100) is None


def test_barra_a_partir_da_quarta_no_mesmo_minuto():
    limite = Limite(por_ip_minuto=3)
    for _ in range(3):
        limite.checar("1.1.1.1", agora=100)
    assert "Muitas perguntas" in limite.checar("1.1.1.1", agora=100)


def test_libera_depois_que_o_minuto_passa():
    limite = Limite(por_ip_minuto=3)
    for _ in range(3):
        limite.checar("1.1.1.1", agora=100)
    assert limite.checar("1.1.1.1", agora=161) is None


def test_um_ip_nao_bloqueia_o_outro():
    limite = Limite(por_ip_minuto=2)
    limite.checar("1.1.1.1", agora=100)
    limite.checar("1.1.1.1", agora=100)
    assert limite.checar("2.2.2.2", agora=100) is None


def test_cota_diaria_vale_para_todos_os_ips():
    limite = Limite(por_ip_minuto=100, global_dia=2)
    limite.checar("1.1.1.1", agora=100)
    limite.checar("2.2.2.2", agora=100)
    assert "cota diária" in limite.checar("3.3.3.3", agora=100)


def test_cota_diaria_reabre_no_dia_seguinte():
    limite = Limite(por_ip_minuto=100, global_dia=2)
    limite.checar("1.1.1.1", agora=100)
    limite.checar("1.1.1.1", agora=100)
    assert limite.checar("1.1.1.1", agora=100 + 86401) is None


if __name__ == "__main__":
    for nome, funcao in list(globals().items()):
        if nome.startswith("test_"):
            funcao()
            print(f"OK  {nome}")
    print("\nTudo certo.")
