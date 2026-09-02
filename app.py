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
from src.agent import load_agent, ask, formatar_fontes, DATA_BASE

# ----------------------------------------------------------------------
# Configuracao geral da pagina (titulo da aba, icone, layout)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Assistente Empresarial",
    page_icon="🌿",
    layout="centered",
)


def caixa_de_fontes(fontes):
    """Desenha as fontes (com categoria) numa caixinha verde discreta.

    'fontes' e a lista devolvida por formatar_fontes():
    [{"categoria": ..., "arquivo": ..., "paginas": [...]}, ...]
    """
    if not fontes:
        return
    linhas = []
    for f in fontes:
        paginas = (
            " · pág. " + ", ".join(map(str, f["paginas"])) if f["paginas"] else ""
        )
        linhas.append(
            f'<span style="opacity:.75">[{f["categoria"]}]</span> '
            f'{f["arquivo"]}{paginas}'
        )
    texto_fontes = "<br>".join(linhas)
    st.markdown(
        f"""
        <div style="
            background: #EAF4EE;
            border-left: 4px solid #1D9E75;
            border-radius: 6px;
            padding: 8px 12px;
            margin-top: 8px;
            font-size: 0.85rem;
            color: #16281F;">
            <strong>📚 Fontes consultadas</strong><br>{texto_fontes}
        </div>
        """,
        unsafe_allow_html=True,
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
st.title("🌿 Assistente Empresarial")
st.caption(
    f"Dúvidas de **direito empresarial** respondidas com base em legislação e "
    f"documentos oficiais brasileiros · base atualizada em {DATA_BASE}."
)

# Aviso claro de que e uma IA (boa pratica e pedido do enunciado)
st.warning(
    "**Conteúdo informativo gerado por IA.** Não constitui consultoria jurídica "
    "ou contábil. Para o seu caso concreto, consulte um advogado ou contador.",
    icon="⚠️",
)

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
            caixa_de_fontes(msg["fontes"])

# ----------------------------------------------------------------------
# Chips de perguntas sugeridas: aparecem so no comeco (sem conversa ainda).
# Ao clicar, a pergunta e "enviada" como se o usuario tivesse digitado.
# ----------------------------------------------------------------------
PERGUNTAS_SUGERIDAS = [
    "Qual a diferença entre MEI, ME e EPP?",
    "O que define juridicamente um empresário?",
    "Quais documentos preciso para abrir uma sociedade limitada?",
    "Qual o prazo para entregar a declaração anual do MEI?",
]

pergunta_do_chip = None
if not st.session_state.mensagens:
    st.markdown("**Comece por uma dúvida comum:**")
    colunas = st.columns(2)
    for i, sugestao in enumerate(PERGUNTAS_SUGERIDAS):
        # distribui os botoes em 2 colunas
        if colunas[i % 2].button(sugestao, key=f"chip_{i}", use_container_width=True):
            pergunta_do_chip = sugestao

# ----------------------------------------------------------------------
# Campo de pergunta (fica fixo no rodape, estilo chat)
# ----------------------------------------------------------------------
pergunta_digitada = st.chat_input("Ex: Qual a diferença entre MEI, ME e EPP?")

# A pergunta pode vir do campo de texto OU de um chip clicado
pergunta = pergunta_digitada or pergunta_do_chip

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

            # Fontes vem dos METADADOS do indice (categoria + arquivo + pagina)
            fontes = formatar_fontes(resultado["source_documents"])

        st.markdown(resposta)
        caixa_de_fontes(fontes)

    # Guarda a resposta no historico
    st.session_state.mensagens.append(
        {"papel": "assistant", "texto": resposta, "fontes": fontes}
    )

# ----------------------------------------------------------------------
# Rodape institucional com links (LinkedIn e GitHub)
# ----------------------------------------------------------------------
LINKEDIN_URL = "https://www.linkedin.com/in/luís-felipe-barbosa-ferreira-33a0a9210"
GITHUB_URL = "https://github.com/oLuisFelipeFerreira/alura-agent"
COR_RODAPE = "#2F6B4F"  # verde-floresta: mais escuro que os botoes, mas legivel

# Icones desenhados em SVG (nao dependem de arquivo de imagem externo)
icone_linkedin = f"""<svg width="20" height="20" viewBox="0 0 24 24" fill="{COR_RODAPE}">
<path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.63-1.85 3.36-1.85 3.6 0 4.27 2.37 4.27 5.45v6.29zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.8 0 0 .78 0 1.74v20.52C0 23.22.8 24 1.77 24h20.45c.98 0 1.78-.78 1.78-1.74V1.74C24 .78 23.2 0 22.22 0z"/>
</svg>"""

icone_github = f"""<svg width="20" height="20" viewBox="0 0 24 24" fill="{COR_RODAPE}">
<path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.39.6.11.82-.26.82-.58v-2.03c-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.33-1.76-1.33-1.76-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.5.99.11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.66.24 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.82 1.1.82 2.22v3.29c0 .32.22.7.83.58C20.56 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z"/>
</svg>"""

st.markdown(
    f"""
    <hr style="margin-top: 2rem; border: none; border-top: 1px solid #D4E6DB;">
    <div style="text-align: center; font-size: 0.85rem; color: {COR_RODAPE}; margin-bottom: 6px;">
        Desenvolvido por <strong>Luís Felipe Ferreira</strong>
    </div>
    <div style="display: flex; justify-content: center; gap: 18px; margin-bottom: 10px;">
        <a href="{LINKEDIN_URL}" target="_blank" title="LinkedIn">{icone_linkedin}</a>
        <a href="{GITHUB_URL}" target="_blank" title="GitHub">{icone_github}</a>
    </div>
    <div style="text-align: center; font-size: 0.75rem; color: #6B7C73;">
        Baseado em documentos oficiais da Receita Federal · Não substitui orientação profissional
    </div>
    """,
    unsafe_allow_html=True,
)
