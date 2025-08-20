import os
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# Imports do projeto (pasta src/)
from src.settings import SETTINGS
from src.milvus_utils import drop_collection
from src.rag import ingest_pdfs, retrieve_top_k
from src.export_pdf import export_chat_pdf
from src.llm_router import EmbeddingsCloud, EmbeddingsLocal, LLMCloud, LLMLocal

# ================== bootstrap ==================
load_dotenv()

st.set_page_config(
    page_title="NUPETR/IDEMA-RN • Chat de Parecer Técnico",
    page_icon="💼",
    layout="wide",
)

st.title("💼 NUPETR/IDEMA-RN — Chat de Parecer Técnico (RAG + Milvus)")
st.caption(
    "Assistente ancorado em manuais internos (PDF). "
    "Preencha os filtros (Tipo de Licença/Empreendimento), envie PDFs e escolha o backend (Nuvem ou Local)."
)

# ================== estado ==================
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []  # (role, text)

# ================== sidebar: chave ==================
st.sidebar.subheader("🔐 Minha chave vs Chave do app")
key_mode = st.sidebar.radio(
    "Escolha a origem da chave OpenAI",
    options=["Minha chave", "Chave do app (secrets)"],
    index=0,
    help="Você pode usar sua própria chave nesta sessão, ou a chave do app definida em Secrets.",
)

user_key = ""
if key_mode == "Minha chave":
    user_key = st.sidebar.text_input(
        "Cole sua chave (formato sk-...)",
        type="password",
        help="Usando a sua chave apenas nesta sessão.",
    )
else:
    # Tenta pegar do secrets
    user_key = st.secrets.get("OPENAI_API_KEY", "")
    if user_key:
        st.sidebar.success("Usando a **chave do app** (secrets).")
    else:
        st.sidebar.warning("Nenhuma chave encontrada em **Secrets**. Você pode alternar para *Minha chave*.")

# Se tiver chave, injeta no SETTINGS para os backends cloud
if user_key:
    os.environ["OPENAI_API_KEY"] = user_key
    SETTINGS.openai_api_key = user_key  # usada por EmbeddingsCloud/LLMCloud

# ================== sidebar: modo do modelo ==================
st.sidebar.subheader("🧠 Modo do modelo")
mode = st.sidebar.radio(
    "Escolha o modo",
    options=["OpenAI (com chave)", "Modelo Local (sem chave)"],
    index=0,
)

# Valida pré-condições
use_cloud = (mode == "OpenAI (com chave)")
if use_cloud and not SETTINGS.openai_api_key:
    st.sidebar.error("Para usar **OpenAI**, informe a chave (Minha chave) ou configure em Secrets.")
    emb = None
    llm = None
else:
    emb = EmbeddingsCloud() if use_cloud else EmbeddingsLocal()   # all-MiniLM-L6-v2 local
    llm = LLMCloud() if use_cloud else LLMLocal()                 # OpenAI / Ollama

# ================== sidebar: filtros ==================
st.sidebar.subheader("Filtros por metadados")
tipo_lic = st.sidebar.selectbox("Tipo de Licença", ["RLO", "LP", "LI", "LO", "OUTROS"], index=0)
tipo_emp = st.sidebar.selectbox("Tipo de Empreendimento", ["POÇO", "ESTAÇÃO", "OLEODUTO", "BASE", "OUTROS"], index=0)

# Expressão Milvus para filtrar via busca vetorial
expr = f'tipo_licenca == "{tipo_lic}" && tipo_empreendimento == "{tipo_emp}"'

# ================== sidebar: PDFs & ações ==================
st.sidebar.subheader("📄 PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Selecione (padrão: tipoLicenca_tipoEmpreendimento.pdf)",
    accept_multiple_files=True,
    type=["pdf"],
    help="Limite ~200MB por PDF.",
)

st.sidebar.subheader("Ações")
if st.sidebar.button("🧹 Limpar histórico"):
    st.session_state.history = []
    st.success("Histórico limpo.")

if st.sidebar.button("🗑️ Clear Collection (Milvus)"):
    try:
        drop_collection(SETTINGS.milvus_collection)
        st.success(f"Coleção '{SETTINGS.milvus_collection}' limpa.")
    except Exception as e:
        st.error(f"Falha ao limpar coleção: {e}")

# Botão principal de ingestão
ing_btn = st.sidebar.button("📥 Indexar PDFs no Milvus", disabled=(emb is None or not uploaded_files))

# ================== ingestão ==================
if ing_btn:
    if emb is None:
        st.error("Defina a chave para usar o modo OpenAI **ou** mude para *Modelo Local (sem chave)*.")
    elif not uploaded_files:
        st.error("Envie pelo menos um PDF.")
    else:
        try:
            # (nome, bytes) para cada arquivo
            pairs = [(f.name, f.read()) for f in uploaded_files]
            n_chunks = ingest_pdfs(
                encoder=emb,
                files=pairs,
                tipo_licenca=tipo_lic,
                tipo_empreendimento=tipo_emp,
                collection_name=SETTINGS.milvus_collection,
            )
            st.success(f"Indexação concluída. {n_chunks} trechos inseridos no Milvus.")
        except Exception as e:
            st.error(f"Falha na indexação: {e}")

# ================== conversa ==================
st.header("Conversa")

# Mostra histórico
for role, msg in st.session_state.history:
    with st.chat_message("assistant" if role == "assistant" else "user"):
        st.markdown(msg)

# Caixa de pergunta
user_q = st.chat_input("Digite sua pergunta aqui...")

if user_q and llm is not None:
    # adiciona pergunta ao feed
    st.session_state.history.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    # Recupera contexto do Milvus com os filtros
    try:
        hits = retrieve_top_k(
            encoder=emb,
            query=user_q,
            collection_name=SETTINGS.milvus_collection,
            top_k=5,
            expr=expr,
        )
    except Exception as e:
        hits = []
        st.error(f"Falha ao buscar no Milvus: {e}")

    # Monta contexto (texto puro) para o LLM
    context_blocks = [h["text"] for h in hits]
    try:
        answer = llm.answer(user_q, context_blocks)
    except Exception as e:
        answer = f"Não foi possível gerar a resposta ({e})."

    # Monta referências
    refs = "\n".join([
        f"• {h['fonte']} (p.{h['pagina']}) — {h['tipo_licenca']}/{h['tipo_empreendimento']}"
        for h in hits
    ])
    final_answer = answer if not hits else f"{answer}\n\n**Fontes consultadas:**\n{refs}"

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.history.append(("assistant", final_answer))

# ================== exportar conversa ==================
st.divider()
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🧾 Exportar conversa (PDF)"):
        try:
            out_path = "conversa_nupetr.pdf"
            export_chat_pdf(out_path, st.session_state.history, logo_path=None)
            with open(out_path, "rb") as f:
                st.download_button(
                    label="Baixar conversa em PDF",
                    data=f,
                    file_name=out_path,
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"Falha ao exportar PDF: {e}")
