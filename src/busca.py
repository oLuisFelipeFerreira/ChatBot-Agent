"""
Busca hibrida: combina busca SEMANTICA (embeddings/FAISS) com busca por
PALAVRA-CHAVE (BM25).

Por que hibrida?
- A busca semantica entende SIGNIFICADO ("faturamento" ~ "receita bruta"),
  mas erra com termos exatos.
- O BM25 acerta o TERMO EXATO ("EPP", "art. 966", "81.000"), mas nao
  entende sinonimos.
Em texto juridico as duas coisas importam, entao usamos as duas e
combinamos os resultados.
"""

import re

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Palavras muito comuns em portugues. Sem filtrar isso, o BM25 pontua alto
# qualquer trecho que contenha "de", "entre", "a" -- e traz lixo.
STOPWORDS_PT = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "dela", "dele",
    "do", "dos", "e", "em", "entre", "era", "essa", "esse", "esta", "este",
    "eu", "foi", "for", "isso", "ja", "la", "lhe", "mais", "mas", "me",
    "mesmo", "meu", "na", "nas", "nao", "não", "nem", "no", "nos", "num",
    "numa", "o", "os", "ou", "para", "pela", "pelo", "por", "qual", "quais",
    "quando", "que", "quem", "se", "sem", "ser", "seu", "sua", "sao", "são",
    "so", "só", "tem", "ter", "um", "uma", "voce", "você", "onde", "qualquer",
    "sobre", "pode", "posso", "preciso", "quero", "ha", "há", "the", "of", "to",
}

# Glossario do dominio: as pessoas usam SIGLAS, a lei escreve POR EXTENSO.
# Sem isso, "EPP" nunca casa com "empresa de pequeno porte" no BM25.
SINONIMOS = {
    "mei": "microempreendedor individual",
    "me": "microempresa",
    "epp": "empresa de pequeno porte",
    "ltda": "sociedade limitada",
    "ei": "empresário individual",
    "cnpj": "cadastro nacional da pessoa jurídica",
    "das": "documento de arrecadação do simples nacional",
    "faturamento": "receita bruta",
    "simei": "sistema de recolhimento em valores fixos mensais",
    "drei": "departamento nacional de registro empresarial e integração",
    "junta": "junta comercial",
    "abrir empresa": "constituição registro empresarial",
}


def expandir_consulta(pergunta: str) -> str:
    """Acrescenta os termos por extenso das siglas presentes na pergunta.

    "diferença entre MEI, ME e EPP"
        -> "diferença entre MEI, ME e EPP microempreendedor individual
            microempresa empresa de pequeno porte"

    Isso faz a pergunta do usuario falar a MESMA lingua dos documentos.
    """
    minuscula = pergunta.lower()
    extras = []
    for sigla, extenso in SINONIMOS.items():
        # \b garante que "me" nao case dentro de "mesmo"
        if re.search(rf"\b{re.escape(sigla)}\b", minuscula):
            extras.append(extenso)
    if not extras:
        return pergunta
    return pergunta + " " + " ".join(extras)


def tokenizar_pt(texto: str) -> list:
    """Quebra o texto em palavras uteis para a busca por palavra-chave.

    1) tudo em minusculas
    2) separa por qualquer caractere que nao seja letra ou numero
       (o range a-zà-ÿ cobre os acentos do portugues)
    3) descarta palavras de 1 letra e stopwords
    """
    texto = texto.lower()
    tokens = [t for t in re.split(r"[^0-9a-zà-ÿ]+", texto) if len(t) > 1]
    return [t for t in tokens if t not in STOPWORDS_PT]


def tokenizar_bm25(texto: str) -> list:
    """Tokenizador usado pelo BM25: expande siglas antes de tokenizar.

    Aplicado tanto aos documentos quanto as perguntas, garantindo que os
    dois lados falem a mesma lingua.
    """
    return tokenizar_pt(expandir_consulta(texto))


# Prompt usado para gerar variacoes da pergunta (ver montar_retriever_hibrido)
PROMPT_MULTIQUERY = PromptTemplate(
    input_variables=["question"],
    template=(
        "Você reformula perguntas para busca em documentos jurídicos brasileiros.\n"
        "Gere 3 versões diferentes da pergunta abaixo, usando a linguagem FORMAL "
        "da legislação (ex: 'faturamento' -> 'receita bruta'; 'EPP' -> 'empresa "
        "de pequeno porte'). Uma por linha, sem numerar, sem explicar.\n\n"
        "Pergunta: {question}"
    ),
)


def montar_retriever_hibrido(vectorstore, chunks, k: int,
                             peso_semantico: float = 0.6,
                             llm=None):
    """Monta o buscador hibrido (BM25 + semantico).

    peso_semantico: quanto a busca por significado pesa no resultado final.
    0.6 = 60% semantico / 40% palavra-chave. Bom ponto de partida para texto
    juridico, onde os termos exatos importam bastante.

    llm: se informado, ativa o MULTI-QUERY -- o modelo reescreve a pergunta
    em 3 versoes com vocabulario juridico e busca com todas. Resolve o caso
    em que o usuario usa siglas e a lei escreve por extenso. Custa uma
    chamada extra de API por pergunta (barata, com o Haiku).
    """
    # Buscador por palavra-chave. O tokenizador ja expande siglas.
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=tokenizar_bm25)
    bm25.k = k

    # Buscador semantico, a partir do indice FAISS
    semantico = vectorstore.as_retriever(search_kwargs={"k": k})

    # O EnsembleRetriever roda os dois e funde os rankings
    hibrido = EnsembleRetriever(
        retrievers=[bm25, semantico],
        weights=[1 - peso_semantico, peso_semantico],
    )

    final = hibrido if llm is None else MultiQueryRetriever.from_llm(
        retriever=hibrido, llm=llm, prompt=PROMPT_MULTIQUERY
    )

    # CORTE FINAL em k trechos.
    #
    # Sem isto, k=10 devolve ate 19: o EnsembleRetriever roda BM25 e
    # semantica, cada um traz 10, e ele entrega a UNIAO dos dois sem cortar.
    # Com multi-query a conta piora, porque sao tres perguntas buscando.
    # Resultado medido: 13.519 caracteres de contexto onde se esperava
    # ~8.200 -- quase o dobro do custo e da espera, sem ninguem ter
    # decidido isso.
    #
    # O corte vem DEPOIS da fusao porque o Ensemble ja ordenou por
    # relevancia combinada: os 10 primeiros sao os melhores dos dois
    # buscadores. Reduzir cada buscador para 5 traria 10 piores.
    return final | RunnableLambda(lambda docs: docs[:k])
