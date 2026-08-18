"""
Interface web (Streamlit) para o agente de MEI/Simples Nacional.

Reaproveita o agente que ja existe (src/agent.py e src/ingest.py):
- constroi o indice FAISS se ainda nao existir;
- carrega a chain de perguntas e respostas;
- mostra uma pagina web com campo de pergunta, resposta e fontes.

Uso:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src.ingest import build_vectorstore, INDEX_DIR
from src.agent import load_agent, ask

# ----------------------------------------------------------------------
# Configuracao geral da pagina (titulo da aba, icone, layout)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Assistente MEI",
    page_icon="📄",
    layout="centered",
)

# ----------------------------------------------------------------------
# Carrega o agente UMA vez e guarda em cache.
# O @st.cache_resource evita reconstruir o indice e recarregar o modelo
# a cada pergunta (senao ficaria lento demais).
# ----------------------------------------------------------------------
@st.cache_resource
def iniciar_agente():
    if not Path(INDEX_DIR).exists():
        build_vectorstore("data")
    return load_agent()


# ----------------------------------------------------------------------
# Cabecalho
# ----------------------------------------------------------------------
st.title("📄 Assistente MEI")
st.caption("Tire dúvidas sobre **MEI** e **Simples Nacional** — respostas baseadas em documentos oficiais da Receita Federal.")

# Aviso claro de que e uma IA (boa pratica e pedido do enunciado)
st.info("Este é um assistente de IA com fins informativos. Não substitui orientação de um contador ou advogado.", icon="ℹ️")

# Prepara o agente (mostra um aviso enquanto carrega na primeira vez)
with st.spinner("Preparando o assistente..."):
    chain = iniciar_agente()

# ----------------------------------------------------------------------
# Historico da conversa: guardamos as mensagens na "sessao" do usuario
# ----------------------------------------------------------------------
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Mostra o historico ja existente
for msg in st.session_state.mensagens:
    with st.chat_message(msg["papel"]):
        st.markdown(msg["texto"])
        if msg.get("fontes"):
            st.caption("📚 Fontes: " + msg["fontes"])

# ----------------------------------------------------------------------
# Campo de pergunta (fica fixo no rodape, estilo chat)
# ----------------------------------------------------------------------
pergunta = st.chat_input("Ex: Qual o prazo para entregar a declaração anual do MEI?")

if pergunta:
    # Mostra a pergunta do usuario
    st.session_state.mensagens.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Gera a resposta
    with st.chat_message("assistant"):
        with st.spinner("Consultando os documentos..."):
            resultado = ask(chain, pergunta)
            resposta = resultado["result"]

            # Monta a lista de fontes (arquivo + pagina)
            fontes = set()
            for doc in resultado["source_documents"]:
                nome = Path(doc.metadata.get("source", "?")).name
                pagina = doc.metadata.get("page")
                if pagina is not None:
                    fontes.add(f"{nome} (pág. {pagina + 1})")
                else:
                    fontes.add(nome)
            texto_fontes = ", ".join(sorted(fontes))

        st.markdown(resposta)
        if texto_fontes:
            st.caption("📚 Fontes: " + texto_fontes)

    # Guarda a resposta no historico
    st.session_state.mensagens.append(
        {"papel": "assistant", "texto": resposta, "fontes": texto_fontes}
    )
