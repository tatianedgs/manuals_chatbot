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
st.caption("Assistente ancorado em manuais internos (PDF). Use filtros por Tipo de Licença/Empreendimento e escolha backend Nuvem ou Local.")

# ===== Estado =====
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []
if "mode" not in st.session_state:
    st.session_state.mode = "Nuvem"

# ===== Sidebar: Config =====
st.sidebar.header("Configuração")
mode = st.sidebar.radio("Backend", ["Nuvem", "Local"], index=(0 if st.session_state.mode=="Nuvem" else 1))
st.session_state.mode = mode

# Embeddings/LLM backend
if mode == "Nuvem":
    emb = EmbeddingsCloud()
    llm = LLMCloud()
    if not SETTINGS.openai_api_key:
        st.sidebar.warning("Defina OPENAI_API_KEY para usar o modo Nuvem.")
else:
    emb = EmbeddingsLocal()  # all-MiniLM-L6-v2
    llm = LLMLocal()         # Ollama (opcional)

# Filtros do domínio
st.sidebar.subheader("Filtros por metadados")
tipo_lic = st.sidebar.selectbox("Tipo de Licença", ["RLO", "LP", "LI", "LO", "OUTROS"], index=0)
tipo_emp = st.sidebar.selectbox("Tipo de Empreendimento", ["POÇO", "ESTAÇÃO", "OLEODUTO", "BASE", "OUTROS"], index=0)

# Expressão Milvus
expr = f' tipo_licenca == "{tipo_lic}" && tipo_empreendimento == "{tipo_emp}" '

# ===== Ações =====
st.sidebar.subheader("Ações")
if st.sidebar.button("🧹 Limpar histórico"):
    st.session_state.history = []
    st.success("Histórico limpo.")

if st.sidebar.button("🗑️ Clear Collection (Milvus)"):
    drop_collection(SETTINGS.milvus_collection)
    st.success("Coleção do Milvus removida.")

if st.sidebar.button("📄 Exportar chat em PDF"):
    out_path = os.path.join(os.getcwd(), "chat_export.pdf")
    export_chat_pdf(out_path, st.session_state.history, logo_path=os.path.join("assets", "logo_idema.png"))
    st.sidebar.success("PDF exportado!")
    st.sidebar.download_button("Baixar PDF", data=open(out_path, "rb").read(), file_name="chat_export.pdf")

st.sidebar.divider()

# ===== Ingestão =====
st.subheader("📥 Ingestão RAG — PDFs Institucionais")
files = st.file_uploader("Envie 1 ou mais PDFs (manuais/modelos/processos)", type=["pdf"], accept_multiple_files=True)
col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("Indexar PDFs no Milvus") and files:
        uploaded = [(f.name, f.read()) for f in files]
        with st.spinner("Gerando embeddings e inserindo no Milvus..."):
            n = ingest_pdfs(
                encoder=emb,
                files=uploaded,
                tipo_licenca=tipo_lic,
                tipo_empreendimento=tipo_emp,
                collection_name=SETTINGS.milvus_collection,
            )
        st.success(f"{n} trechos inseridos na coleção '{SETTINGS.milvus_collection}'.")
with col_b:
    st.info("Os documentos indexados ficam filtráveis por Tipo de Licença/Empreendimento.")

st.divider()

# ===== Chat =====
st.subheader("💬 Chat de Parecer Técnico")

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

user_q = st.chat_input("Faça sua pergunta (ex.: 'Quais itens checar para RLO de poços?')")

if user_q:
    st.session_state.history.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.spinner("Buscando no Milvus e consultando o LLM..."):
        hits = retrieve_top_k(emb, user_q, SETTINGS.milvus_collection, top_k=5, expr=expr)
        ctx = [h["text"] for h in hits]
        answer = llm.answer(user_q, ctx)

        refs = "\n".join([
    f"• {h['fonte']} (p.{h['pagina']}) — {h['tipo_licenca']}/{h['tipo_empreendimento']}"
    for h in hits
])

if hits:
    final_answer = f"{answer}\n\n**Fontes consultadas:**\n{refs}"
else:
    final_answer = answer

with st.chat_message("assistant"):
    st.markdown(final_answer)
st.session_state.history.append(("assistant", final_answer))