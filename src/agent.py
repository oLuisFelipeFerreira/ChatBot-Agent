"""
Monta a chain de recuperacao + resposta (RAG) usando Claude (Anthropic)
como modelo de linguagem e o indice FAISS gerado pelo ingest.py.
"""

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .ingest import INDEX_DIR, EMBEDDING_MODEL, carregar_chunks
from .busca import montar_retriever_hibrido

load_dotenv()

# Quantos trechos recuperar por pergunta.
# A base tem ~2000 chunks (leis + manuais), entao vale recuperar mais
# contexto para o modelo ter chance de achar a resposta.
TOP_K = 10

# Data da ultima atualizacao da base de documentos (exibida ao usuario).
DATA_BASE = "agosto de 2026"

# Multi-query: o modelo reescreve a pergunta em 3 versoes com vocabulario
# juridico antes de buscar. Melhora bastante a recuperacao, MAS gasta uma
# chamada extra de API por pergunta.
# Coloque False para economizar chamadas ou se a API estiver sobrecarregada
# (erro 529). A expansao de siglas do glossario continua funcionando de
# qualquer forma -- ela e local e nao custa nada.
USAR_MULTIQUERY = True

# Prompt que "ancora" o modelo: ele so pode responder com base no contexto
# recuperado, e deve admitir quando a resposta nao estiver ali.
PROMPT_TEMPLATE = """Você é um assistente especializado em Direito Empresarial brasileiro.
Responde dúvidas de empreendedores sobre abertura de empresa, tipos societários,
registro empresarial, obrigações tributárias, MEI e Simples Nacional.

REGRAS OBRIGATÓRIAS:
1. Use SOMENTE as informações do contexto abaixo. Não use conhecimento externo,
   não invente números, prazos ou artigos.
2. NÃO GENERALIZE POR ANALOGIA. Se o contexto afirma algo sobre uma categoria
   (ex: o MEI), não conclua que vale para outra (ex: ME ou EPP). Cada requisito
   só pode ser afirmado se estiver explícito no contexto para AQUELA categoria.
3. Se a resposta não estiver no contexto, responda exatamente:
   "Não encontrei essa informação nos documentos disponíveis."
4. NÃO CONFUNDA AS TRÊS CLASSIFICAÇÕES, que são independentes entre si:
   - TIPO JURÍDICO (como a empresa é constituída): empresário individual,
     sociedade limitada, sociedade limitada unipessoal, sociedade anônima...
   - PORTE (faixa de receita bruta): ME, EPP, ou demais.
   - REGIME TRIBUTÁRIO (como paga impostos): Simples Nacional, Lucro
     Presumido, Lucro Real.
   Uma empresa tem as três coisas ao mesmo tempo, e uma não determina a outra.
   O MEI é a exceção: está obrigatoriamente vinculado ao Simei/Simples Nacional.
5. Quando o contexto citar um artigo de lei, mencione-o na resposta
   (ex: "conforme o art. 966 do Código Civil").
6. ATENÇÃO À VIGÊNCIA: se o contexto contiver notas de atualização indicando que
   algo foi revogado, extinto ou alterado, PREVALECE a informação mais recente.
   Avise o usuário quando a redação de uma lei mencionar instituto já extinto.
7. Escreva em português claro, para quem não é advogado. Explique termos técnicos.
8. Nunca afirme o que o usuário "deve fazer" no caso concreto dele — explique o
   que a norma diz e sugira confirmar com um profissional quando o caso for específico.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)


def load_agent(index_dir: str = INDEX_DIR):
    """Carrega o indice salvo e monta a chain de perguntas e respostas."""
    if not Path(index_dir).exists():
        raise FileNotFoundError(
            f"Indice '{index_dir}' nao encontrado. Rode antes: python -m src.ingest"
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY nao encontrada. Copie .env.example para .env "
            "e preencha com sua chave."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )

    # Busca hibrida: palavra-chave (BM25) + semantica (FAISS).
    # Se os chunks salvos nao existirem (indice antigo), cai para a busca
    # semantica pura, avisando -- assim o app nao quebra.
    # claude-haiku-4-5: modelo rapido e economico.
    # max_retries: a API pode responder 529 (sobrecarga temporaria) ou 429
    # (limite de taxa). Sao erros passageiros, entao a propria biblioteca
    # tenta de novo, com espera crescente entre as tentativas.
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        max_retries=5,
        timeout=60,
    )

    chunks = carregar_chunks()
    if chunks:
        retriever = montar_retriever_hibrido(
            vectorstore,
            chunks,
            k=TOP_K,
            llm=llm if USAR_MULTIQUERY else None,
        )
    else:
        print(
            "AVISO: chunks.json nao encontrado -- usando apenas busca semantica.\n"
            "       Reindexe (apague faiss_index/) para ativar a busca hibrida."
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def ask(chain, pergunta: str) -> dict:
    """Faz uma pergunta ao agente.

    Devolve um dicionario com a resposta ('result') e os trechos de
    documento usados ('source_documents'), para citar a fonte.
    """
    return chain.invoke({"query": pergunta})


# Frase exata que o prompt manda o modelo usar quando nao acha a resposta.
FRASE_SEM_RESPOSTA = "Não encontrei essa informação nos documentos disponíveis"


def sem_resposta(texto: str) -> bool:
    """Detecta se o agente respondeu que nao encontrou a informacao.

    Nesse caso nao faz sentido listar fontes: nenhum documento embasou a
    resposta, e mostrar 10 arquivos passaria a impressao errada.
    """
    return FRASE_SEM_RESPOSTA.lower() in texto.lower()


def formatar_fontes(source_documents) -> list:
    """Extrai as fontes dos metadados, agrupadas e sem repeticao.

    Retorna uma lista de dicts: {"categoria": ..., "arquivo": ..., "paginas": [...]}
    A citacao vem dos METADADOS do indice, nao do texto gerado pelo modelo --
    assim o numero da pagina nunca sai errado.
    """
    agrupado = {}
    for doc in source_documents:
        arquivo = doc.metadata.get("arquivo") or Path(
            doc.metadata.get("source", "?")
        ).name
        categoria = doc.metadata.get("categoria", "Geral")
        pagina = doc.metadata.get("page")

        chave = (categoria, arquivo)
        agrupado.setdefault(chave, set())
        if pagina is not None:
            agrupado[chave].add(pagina + 1)

    fontes = []
    for (categoria, arquivo), paginas in sorted(agrupado.items()):
        fontes.append(
            {
                "categoria": categoria,
                "arquivo": arquivo,
                "paginas": sorted(paginas),
            }
        )
    return fontes
