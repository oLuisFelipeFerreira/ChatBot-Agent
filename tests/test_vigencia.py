"""
Teste da camada de vigencia: as notas de curadoria chegam ao modelo?

Por que este teste existe: as notas sao 5 trechos numa base de 2.099
(0,24%). Perguntado sobre EIRELI, o buscador comum nao trazia NENHUM
deles -- perdiam todas as vagas para os 52 trechos de outros documentos
que citam a EIRELI como opcao valida. O agente respondia com a redacao
vencida, com total confianca.

E uma falha silenciosa: nada quebra, a resposta parece boa, e so quem
conhece o assunto percebe o erro. Por isso precisa de teste automatico.

Uso:
    python test_vigencia.py
"""

from langchain_core.documents import Document

from src.busca import montar_retriever_com_vigencia, ARQUIVO_VIGENCIA


def doc(texto, arquivo):
    return Document(page_content=texto, metadata={"arquivo": arquivo})


# Reproduz a proporcao real da base: a nota correta e minoria esmagadora.
NOTA = doc("A EIRELI foi EXTINTA pela Lei 14.195/2021.", ARQUIVO_VIGENCIA)
VOLUMOSOS = [
    doc(f"A EIRELI exige capital minimo. Paragrafo {i}.", "in_drei_81.pdf")
    for i in range(40)
]


class BuscadorFalso:
    """Simula o buscador comum, que so encontra os documentos volumosos."""
    def __init__(self, resultado):
        self.resultado = resultado

    def invoke(self, _pergunta):
        return list(self.resultado)


def test_a_nota_entra_mesmo_quando_o_buscador_comum_nao_a_traz():
    principal = BuscadorFalso(VOLUMOSOS[:4])
    com_vigencia = montar_retriever_com_vigencia(principal, [NOTA] + VOLUMOSOS)
    arquivos = [d.metadata["arquivo"] for d in com_vigencia.invoke("Posso abrir uma EIRELI?")]
    assert ARQUIVO_VIGENCIA in arquivos


def test_a_nota_vem_primeiro():
    """O prompt manda a vigencia prevalecer; o modelo pesa mais o inicio."""
    principal = BuscadorFalso(VOLUMOSOS[:4])
    com_vigencia = montar_retriever_com_vigencia(principal, [NOTA] + VOLUMOSOS)
    primeiro = com_vigencia.invoke("EIRELI")[0]
    assert primeiro.metadata["arquivo"] == ARQUIVO_VIGENCIA


def test_nao_duplica_quando_o_buscador_comum_ja_trouxe_a_nota():
    principal = BuscadorFalso([NOTA] + VOLUMOSOS[:3])
    com_vigencia = montar_retriever_com_vigencia(principal, [NOTA] + VOLUMOSOS)
    docs = com_vigencia.invoke("EIRELI")
    assert sum(1 for d in docs if d.metadata["arquivo"] == ARQUIVO_VIGENCIA) == 1


def test_sem_notas_na_base_nao_quebra():
    """Base sem o arquivo de curadoria: avisa, mas continua funcionando."""
    principal = BuscadorFalso(VOLUMOSOS[:4])
    igual = montar_retriever_com_vigencia(principal, VOLUMOSOS)
    assert len(igual.invoke("EIRELI")) == 4


if __name__ == "__main__":
    for nome, funcao in list(globals().items()):
        if nome.startswith("test_"):
            funcao()
            print(f"OK  {nome}")
    print("\nTudo certo.")
