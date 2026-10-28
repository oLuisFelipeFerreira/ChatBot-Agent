"""
Roda todos os testes. Sem framework, sem dependencia, menos de 1 segundo.

    python -m tests.rodar

Cada teste existe por causa de um defeito que aconteceu de verdade; o
docstring de cada arquivo conta qual.
"""

import importlib
import sys

MODULOS = [
    "tests.test_ancoragem",
    "tests.test_vigencia",
    "tests.test_memoria",
    "tests.test_limite",
    "tests.test_provedor",
]


def main() -> int:
    falhas = 0
    for nome in MODULOS:
        modulo = importlib.import_module(nome)
        print(f"\n{nome.split('.')[-1]}")
        for atributo in vars(modulo):
            if not atributo.startswith("test_"):
                continue
            try:
                getattr(modulo, atributo)()
                print(f"  ok    {atributo}")
            except AssertionError as erro:
                falhas += 1
                print(f"  FALHA {atributo}: {erro}")
    print("\nTudo certo." if not falhas else f"\n{falhas} falha(s).")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
