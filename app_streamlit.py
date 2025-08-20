import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Tuple

from src.settings import SETTINGS
from src.milvus_utils import drop_collection
from src.rag import ingest_pdfs, retrieve_top_k
from src.export_pdf import export_chat_pdf
from src.llm_router import EmbeddingsCloud, EmbeddingsLocal, LLMCloud, LLMLocal

load_dotenv()
st.set_page_config(page_title="NUPETR/IDEMA-RN • Chat de Parecer Técnico", page_icon="💼", layout="wide")
st.title("💼 NUPETR/IDEMA-RN — Chat de Parecer Técnico (LLM + RAG + Milvus)")
st.caption("Assistente ancorado em manuais internos (PDF). Preencha os filtros (Tipo de Licença/Empreendimento), envie PDFs e escolha o backend (Nuvem ou Local).")

# ===== Estado =====
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []
if "mode" not in st.session_state:
    st.session_state.mode = "Nuvem"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ===== Sidebar: Config =====
st.sidebar.header("Configuração")

# Campo para o usuário digitar a própria chave (só vale nesta sessão)
st.sidebar.write("**Minha chave**")
user_key = st.sidebar.text_input("Cole sua chave (formato sk-...)", type="password", value=st.session_state.api_key)
st.session_state.api_key = user_key.strip()

mode = st.sidebar.radio("Escolha o modo", ["OpenAI (com chave)", "Modelo Local (sem chave)"], index=(0 if st.session_state.mode == "Nuvem" else 1))
st.session_state.mode = "Nuvem" if mode.startswith("OpenAI") else "Local"

# Inicializa backends conforme o modo
emb = None
llm = None
if st.session_state.mode == "Nuvem":
    if not st.session_state.api_key and not SETTINGS.openai_api_key:
        st.sidebar.error("Informe sua OPENAI_API_KEY na caixa acima (ou configure nos Secrets) para usar o modo Nuvem.")
    else:
        emb = EmbeddingsCloud(api_key=(st.session_state.api_key or None))
        llm = LLMCloud(api_key=(st.session_state.api_key or None))
else:
    emb = EmbeddingsLocal()   # CPU
    llm = LLMLocal()          # Ollama (opcional)

# ===== Filtros do domínio =====
st.sidebar.subheader("Filtros por metadados")
tipo_lic = st.sidebar.text_input("Tipo de Licença", value="RLO")
tipo_emp = st.sidebar.text_input("Tipo de Empreendimento", value="POÇO")

# Expressão Milvus
expr = f'tipo_licenca == "{tipo_lic}" && tipo_empreendimento == "{tipo_emp}"'

# ===== Ações rápidas =====
st.sidebar.subheader("Ações")
if st.sidebar.button("🧹 Limpar histórico"):
    st.session_state.history = []
    st.success("Histórico limpo.")

if st.sidebar.button("🗑️ Limpar coleção (Milvus)"):
    try:
        drop_collection(SETTINGS.milvus_collection)
        st.success("Coleção do Milvus
