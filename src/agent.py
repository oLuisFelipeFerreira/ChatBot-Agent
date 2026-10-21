"""
Monta a chain de recuperacao + resposta (RAG) sobre o indice FAISS gerado
pelo ingest.py.

O modelo de linguagem e trocavel: veja PROVEDOR/MODELO abaixo e o .env.example.
"""

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from .ingest import INDEX_DIR, EMBEDDING_MODEL, carregar_chunks
from .busca import montar_retriever_hibrido

load_dotenv()

# Quantos trechos recuperar por pergunta.
# A base tem ~2000 chunks (leis + manuais), entao vale recuperar mais
# contexto para o modelo ter chance de achar a resposta.
#
# Mas este numero e o que mais pesa na ESPERA. Medido: o modelo gasta a
# maior parte do tempo LENDO o contexto, nao escrevendo a resposta --
# 74s de leitura para 3s de escrita, num modelo local. Cada trecho a
# menos corta a espera quase na mesma proporcao.
#
# Por isso e configuravel: 10 onde o modelo e rapido, menos onde nao e.
# Se 10 responde melhor que 4 e pergunta em aberto -- quem decide isso
# com numero, e nao com palpite, e o conjunto de avaliacao.
TOP_K = int(os.getenv("TOP_K", "10"))

# Qual modelo de linguagem usar. Trocavel pelo .env, sem mexer no codigo:
#   LLM_PROVIDER=ollama        LLM_MODEL=llama3.1   -> local, sem chave
#   LLM_PROVIDER=groq          LLM_MODEL=llama-3.3-70b-versatile
#   LLM_PROVIDER=anthropic     LLM_MODEL=claude-haiku-4-5-20251001
# Cada provedor exige seu pacote (langchain-ollama, langchain-groq...).
# Veja .env.example.
#
# O padrao e o OLLAMA, que roda na sua propria maquina: quem clonar este
# repositorio consegue testar sem chave, sem cartao e sem cota. E mais
# lento que uma API e obedece as regras do prompt com menos consistencia
# -- o quanto, exatamente, esta medido no README.
#
# Os embeddings NAO dependem desta escolha: rodam local (HuggingFace), de
# graca, e trocar de modelo nao obriga a reindexar nada.
PROVEDOR = os.getenv("LLM_PROVIDER", "ollama")
MODELO = os.getenv("LLM_MODEL", "llama3.2:3b")

# Variavel de ambiente com a chave de cada provedor. Ollama roda na sua
# maquina e nao usa chave nenhuma.
CHAVE_DO_PROVEDOR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "ollama": None,
}

# Data da ultima atualizacao da base de documentos (exibida ao usuario).
DATA_BASE = "agosto de 2026"

# Multi-query: o modelo reescreve a pergunta em 3 versoes com vocabulario
# juridico antes de buscar. Melhora a recuperacao, MAS **dobra o numero de
# chamadas por pergunta** -- uma para reescrever, outra para responder.
#
# Isso importa muito em camada gratuita, onde a cota e contada em
# REQUISICOES por dia: com multi-query ligado, uma cota de 20/dia vira 10
# perguntas. Por isso e configuravel pelo .env: ligado no seu computador,
# onde voce quer a melhor recuperacao; desligado no servidor publico, onde
# voce quer o maior numero de perguntas atendidas.
#
# A expansao de siglas do glossario continua funcionando de qualquer forma
# -- ela e local, instantanea e nao custa chamada nenhuma.
USAR_MULTIQUERY = os.getenv("USAR_MULTIQUERY", "1").lower() in {"1", "true", "sim"}

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

9. Se a pergunta se apoiar no que ja foi conversado ("e para ME?", "e se eu
   passar disso?"), entenda a referencia pelo historico, mas continue
   obedecendo a regra 1: so afirme o que estiver no contexto abaixo.

Contexto:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", PROMPT_TEMPLATE),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Quantas trocas (pergunta + resposta) do historico entram no prompt.
# Mais do que isso encarece cada chamada sem melhorar a continuidade: em
# duvida juridica, o fio da conversa raramente passa de duas ou tres voltas.
MAX_HISTORICO = 4

# Antes de buscar, o modelo reescreve a pergunta como se ela fosse unica.
# "e se eu passar do limite?" vira "o que acontece se o MEI ultrapassar o
# limite de faturamento?". Sem isso, o buscador recebe uma frase sem sujeito
# e traz lixo -- nenhuma busca, por melhor que seja, acha o que nao foi dito.
PROMPT_CONTEXTUALIZAR = ChatPromptTemplate.from_messages([
    ("system",
     "Dado o historico da conversa e a ultima pergunta do usuario, reescreva "
     "a pergunta de forma que ela seja compreensivel SOZINHA, sem o "
     "historico. Resolva pronomes e referencias implicitas usando os termos "
     "da conversa. NAO responda a pergunta, NAO explique: devolva apenas a "
     "pergunta reescrita. Se ela ja for independente, devolva-a como esta."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def montar_historico(historico) -> list:
    """Converte [{'pergunta':..., 'resposta':...}] em mensagens do LangChain.

    Corta as trocas mais antigas, mantendo as ultimas MAX_HISTORICO -- o fim
    da conversa e o que da sentido a pergunta atual.
    """
    if not historico:
        return []
    mensagens = []
    for troca in historico[-MAX_HISTORICO:]:
        pergunta = (troca.get("pergunta") or "").strip()
        resposta = (troca.get("resposta") or "").strip()
        if pergunta:
            mensagens.append(HumanMessage(content=pergunta))
        if resposta:
            mensagens.append(AIMessage(content=resposta))
    return mensagens


def _conferir_ollama():
    """Confere que o Ollama esta rodando e que o modelo foi baixado.

    Sem isto, quem clona o projeto descobre o problema so ao perguntar, na
    forma de um erro de conexao sem contexto. Como o Ollama e o caminho
    padrao, esta e a falha MAIS provavel de quem esta comecando -- merece
    uma mensagem que diga o que fazer, nao um traceback.
    """
    import json
    import urllib.request

    url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as r:
            baixados = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        raise EnvironmentError(
            "\n".join([
                f"O Ollama nao respondeu em {url}.",
                "  1. Instale: https://ollama.com/download",
                "  2. Deixe-o rodando (ele sobe sozinho apos instalar)",
                f"  3. Baixe o modelo: ollama pull {MODELO}",
                "Ou use uma API no lugar dele: veja .env.example.",
            ])
        ) from None

    # "llama3.2:3b" e "llama3.2" apontam para o mesmo modelo baixado.
    if not any(m == MODELO or m.split(":")[0] == MODELO.split(":")[0] for m in baixados):
        tem = ", ".join(baixados) if baixados else "nenhum"
        raise EnvironmentError(
            "\n".join([
                f"O modelo '{MODELO}' nao foi baixado ainda.",
                f"  Rode: ollama pull {MODELO}",
                f"  (modelos que voce ja tem: {tem})",
            ])
        )


def criar_llm():
    """Cria o modelo de linguagem escolhido em LLM_PROVIDER / LLM_MODEL.

    init_chat_model importa o pacote do provedor na hora, entao o projeto
    nao precisa depender de todos eles de uma vez.
    """
    if PROVEDOR == "ollama":
        _conferir_ollama()

    chave = CHAVE_DO_PROVEDOR.get(PROVEDOR, f"{PROVEDOR.upper()}_API_KEY")
    if chave and not os.getenv(chave):
        raise EnvironmentError(
            f"{chave} nao encontrada (LLM_PROVIDER={PROVEDOR}). Copie "
            ".env.example para .env e preencha a chave -- ou troque o "
            "provedor la (ollama roda local e nao pede chave)."
        )

    if PROVEDOR == "ollama":
        # num_ctx: a janela de contexto do Ollama e PEQUENA por padrao.
        # Este projeto manda ~3.500 tokens (10 trechos + as regras do
        # prompt). Se a janela for menor, o Ollama CORTA o prompt em
        # silencio -- sem erro e sem aviso -- e o modelo responde com
        # metade dos documentos. O sintoma e "o modelo local e burro",
        # quando na verdade ele nunca recebeu o texto inteiro.
        # 8192 cobre o contexto com folga para a resposta.
        extras = {"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))}
    else:
        # APIs remotas respondem 429 (limite de taxa) ou 529 (sobrecarga)
        # de vez em quando; sao passageiros e a biblioteca tenta de novo.
        extras = {"max_retries": 5, "timeout": 60}

    try:
        return init_chat_model(MODELO, model_provider=PROVEDOR, **extras)
    except ImportError as erro:
        raise ImportError(
            f"Falta o pacote do provedor '{PROVEDOR}'. Instale-o "
            f"(ex.: pip install langchain-{PROVEDOR.replace('_', '-')}) "
            "ou troque LLM_PROVIDER no .env."
        ) from erro


def load_agent(index_dir: str = INDEX_DIR):
    """Carrega o indice salvo e monta a chain de perguntas e respostas."""
    if not Path(index_dir).exists():
        raise FileNotFoundError(
            f"Indice '{index_dir}' nao encontrado. Rode antes: python -m src.ingest"
        )

    # Criado antes dos embeddings: se faltar chave ou pacote do provedor,
    # falha em 1 segundo em vez de depois de carregar o modelo de embeddings.
    llm = criar_llm()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )

    # Busca hibrida: palavra-chave (BM25) + semantica (FAISS).
    # Se os chunks salvos nao existirem (indice antigo), cai para a busca
    # semantica pura, avisando -- assim o app nao quebra.

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

    # Duas partes:
    #   - responder: recebe os trechos ja encontrados e escreve a resposta
    #   - create_retrieval_chain: busca os trechos e passa para o responder
    # Separar as duas e o que permite, adiante, reescrever a pergunta com o
    # historico antes de buscar, e transmitir a resposta em pedacos.
    # Reescreve a pergunta com base no historico ANTES de buscar.
    # Detalhe que importa para a cota: o LangChain so chama o modelo para
    # reescrever quando ha historico. Primeira pergunta da conversa nao
    # paga nada a mais.
    retriever_com_historico = create_history_aware_retriever(
        llm, retriever, PROMPT_CONTEXTUALIZAR
    )

    responder = create_stuff_documents_chain(llm, PROMPT)
    return create_retrieval_chain(retriever_com_historico, responder)


def ask(chain, pergunta: str, historico=None) -> dict:
    """Faz uma pergunta ao agente.

    Devolve um dicionario com a resposta ('result') e os trechos de
    documento usados ('source_documents'), para citar a fonte.

    A cadeia moderna devolve as chaves 'answer' e 'context'; traduzimos
    aqui para os nomes que o resto do projeto ja usa. Assim a troca da
    cadeia nao obriga a mexer na API, no terminal e no Streamlit de uma vez.
    """
    resultado = chain.invoke({
        "input": pergunta,
        "chat_history": montar_historico(historico),
    })
    return {
        "result": resultado["answer"],
        "source_documents": resultado["context"],
    }


# Frase exata que o prompt manda o modelo usar quando nao acha a resposta.
FRASE_SEM_RESPOSTA = "Não encontrei essa informação nos documentos disponíveis"


def truncar_na_recusa(texto: str) -> str:
    """Se o agente recusou, devolve SO a frase de recusa.

    Por que isto existe: observado com um modelo local de 3B, o agente
    respondeu a frase de recusa corretamente e, no paragrafo seguinte,
    inventou um tributo que nao existe ("Imposto Municipal Sobre a
    Renda"). A recusa estava certa; o que veio depois era alucinacao.

    O prompt manda responder EXATAMENTE a frase quando nao sabe, entao
    qualquer continuacao ja esta fora do contrato. Num assistente
    juridico, perder uma nuance eventual e muito mais barato do que
    exibir um imposto inventado -- por isso cortamos.

    Vale para QUALQUER modelo: o grande erra menos, nao erra nunca.
    """
    if not sem_resposta(texto):
        return texto
    return FRASE_SEM_RESPOSTA + "."


def sem_resposta(texto: str) -> bool:
    """Detecta se o agente respondeu que nao encontrou a informacao.

    Nesse caso nao faz sentido listar fontes: nenhum documento embasou a
    resposta, e mostrar 10 arquivos passaria a impressao errada.
    """
    return FRASE_SEM_RESPOSTA.lower() in texto.lower()


def formatar_fontes(source_documents) -> list:
    """Extrai as fontes dos metadados, agrupadas e sem repeticao.

    Retorna uma lista de dicts:
        {"categoria": ..., "arquivo": ..., "paginas": [...],
         "trechos": [{"pagina": 10, "texto": "..."}, ...]}

    A citacao vem dos METADADOS do indice, nao do texto gerado pelo modelo
    -- assim o numero da pagina nunca sai errado.

    Os TRECHOS sao o texto exato que o modelo recebeu. Eles ja estavam na
    memoria e eram descartados; devolve-los deixa o usuario conferir a
    resposta contra a fonte, em vez de so acreditar. O texto nao e cortado:
    truncar a prova derrota o proposito de mostra-la.
    """
    agrupado = {}
    for doc in source_documents:
        arquivo = doc.metadata.get("arquivo") or Path(
            doc.metadata.get("source", "?")
        ).name
        categoria = doc.metadata.get("categoria", "Geral")
        pagina = doc.metadata.get("page")

        chave = (categoria, arquivo)
        dados = agrupado.setdefault(chave, {"paginas": set(), "trechos": []})
        if pagina is not None:
            dados["paginas"].add(pagina + 1)

        texto = (doc.page_content or "").strip()
        # A busca hibrida roda BM25 e semantica em paralelo e as duas podem
        # trazer o MESMO trecho; o multi-query multiplica isso. Sem remover
        # a repeticao, o usuario veria o mesmo paragrafo tres vezes.
        if texto and all(t["texto"] != texto for t in dados["trechos"]):
            dados["trechos"].append(
                {"pagina": pagina + 1 if pagina is not None else None,
                 "texto": texto}
            )

    fontes = []
    for (categoria, arquivo), dados in sorted(agrupado.items()):
        fontes.append(
            {
                "categoria": categoria,
                "arquivo": arquivo,
                "paginas": sorted(dados["paginas"]),
                "trechos": dados["trechos"],
            }
        )
    return fontes
