from __future__ import annotations

import os
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from src.settings import SETTINGS
from src.milvus_utils import drop_collection
from src.rag import ingest_pdfs, retrieve_top_k
from src.export_pdf import export_chat_pdf

# Backends (protege modo local no Cloud)
try:
    from src.llm_router import EmbeddingsCloud, EmbeddingsLocal, LLMCloud, LLMLocal
    _LOCAL_OK = True
except Exception:
    from src.llm_router import EmbeddingsCloud, LLMCloud
    EmbeddingsLocal = None  # type: ignore
    LLMLocal = None  # type: ignore
    _LOCAL_OK = False

load_dotenv()
st.set_page_config(page_title="NUPETR/IDEMA-RN • Chat de Parecer Técnico", page_icon="💼", layout="wide")
st.title("💼 NUPETR/IDEMA-RN — Chat de Parecer Técnico (RAG + Milvus)")
st.caption("Assistente ancorado em manuais internos (PDF). Preencha os filtros, envie PDFs e escolha o backend (Nuvem ou Local).")

if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []

# ---------------- Sidebar: chaves e modo ----------------
st.sidebar.header("🔐 Minha chave vs Chave do app")
key_mode = st.sidebar.radio("Origem da chave OpenAI", ["Minha chave", "Chave do app (secrets)"], index=0)

user_key = ""
if key_mode == "Minha chave":
    user_key = st.sidebar.text_input("Cole sua chave (sk-...)", type="password")
    if user_key.strip():
        SETTINGS.openai_api_key = user_key.strip()
        os.environ["OPENAI_API_KEY"] = SETTINGS.openai_api_key
else:
    secret_key = st.secrets.get("OPENAI_API_KEY", "")
    if secret_key:
        SETTINGS.openai_api_key = secret_key
        os.environ["OPENAI_API_KEY"] = secret_key
        st.sidebar.success("Usando a **chave do app** (secrets).")
    else:
        st.sidebar.warning("Nenhuma OPENAI_API_KEY definida nos Secrets.")

st.sidebar.subheader("🧠 Modo do modelo")
if _LOCAL_OK:
    mode = st.sidebar.radio("Escolha o modo", ["OpenAI (com chave)", "Modelo Local (sem chave)"], index=0)
else:
    mode = st.sidebar.radio("Escolha o modo", ["OpenAI (com chave)"], index=0)
    st.sidebar.info("Modelo Local indisponível neste ambiente.")

emb = None
llm = None
if mode == "OpenAI (com chave)":
    if not SETTINGS.openai_api_key:
        st.sidebar.error("Informe uma OPENAI_API_KEY para usar o modo OpenAI (cole acima ou configure em Secrets).")
    else:
        emb = EmbeddingsCloud()
        llm = LLMCloud()
else:
    emb = EmbeddingsLocal()  # type: ignore
    llm = LLMLocal()  # type: ignore

# ---------------- Filtros ----------------
st.sidebar.subheader("Filtros por metadados")
tipo_lic = st.sidebar.selectbox("Tipo de Licença", ["RLO", "LP", "LI", "LO", "OUTROS"], index=0)
tipo_emp = st.sidebar.selectbox("Tipo de Empreendimento", ["POÇO", "ESTAÇÃO", "OLEODUTO", "BASE", "OUTROS"], index=0)
expr = f'tipo_licenca == "{tipo_lic}" && tipo_empreendimento == "{tipo_emp}"'

# ---------------- PDFs & ações ----------------
st.sidebar.subheader("📄 PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Selecione (padrão: tipoLicenca_tipoEmpreendimento.pdf)",
    accept_multiple_files=True,
    type=["pdf"],
)
st.sidebar.subheader("Ações")
if st.sidebar.button("🧹 Limpar histórico"):
    st.session_state.history = []
    st.success("Histórico limpo.")

if st.sidebar.button("🗑️ Clear Collection (Milvus)"):
    try:
        drop_collection(SETTINGS.milvus_collection)
        st.success(f"Coleção '{SETTINGS.milvus_collection}' removida.")
    except Exception as e:
        st.error(f"Falha ao remover coleção: {e}")

# (debug) mostrar o URI lido
st.sidebar.caption(f"Milvus URI em uso: {SETTINGS.milvus_uri}")
st.sidebar.caption("Milvus token carregado: " + ("✅" if len(SETTINGS.milvus_token) > 10 else "❌"))


# ---------------- Ingestão ----------------
st.subheader("📥 Ingestão RAG — PDFs Institucionais")
col_i, col_hint = st.columns([1, 1])

with col_i:
    disable_ing = emb is None or (mode.startswith("OpenAI") and not SETTINGS.openai_api_key)
    if st.button("📌 Indexar PDFs no Milvus", disabled=disable_ing or not uploaded_files):
        if not uploaded_files:
            st.warning("Envie pelo menos um PDF.")
        else:
            pairs = [(f.name, f.read()) for f in uploaded_files]
            with st.spinner("Gerando embeddings e inserindo no Milvus..."):
                try:
                    n = ingest_pdfs(
                        encoder=emb,  # type: ignore
                        files=pairs,
                        tipo_licenca=tipo_lic,
                        tipo_empreendimento=tipo_emp,
                        collection_name=SETTINGS.milvus_collection,
                    )
                    st.success(f"Indexação concluída: {n} trechos inseridos.")
                except Exception as e:
                    st.error(f"Falha na indexação: {e}")

with col_hint:
    st.info("Os documentos indexados ficam filtráveis por Tipo de Licença/Empreendimento.")

st.divider()

# ---------------- Conversa ----------------
st.subheader("💬 Conversa")
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

question = st.chat_input("Digite sua pergunta aqui...")
if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    final_answer = ""
    try:
        hits = retrieve_top_k(
            encoder=emb,  # type: ignore
            query=question,
            collection_name=SETTINGS.milvus_collection,
            top_k=5,
            expr=expr,
        )
        ctx = [h["text"] for h in hits]
        answer = llm.answer(question, ctx)  # type: ignore
        if hits:
            refs = "\n".join(
                f"• {h['fonte']} (p.{h['pagina']}) — {h['tipo_licenca']}/{h['tipo_empreendimento']}"
                for h in hits
            )
            final_answer = f"{answer}\n\n**Fontes consultadas:**\n{refs}"
        else:
            final_answer = answer
    except Exception as e:
        final_answer = f"Falha ao buscar/gerar resposta: {e}"

    with st.chat_message("assistant"):
        st.markdown(final_answer)
    st.session_state.history.append(("assistant", final_answer))

st.divider()
if st.button("🧾 Exportar conversa (PDF)"):
    try:
        out_path = "conversa_nupetr.pdf"
        export_chat_pdf(out_path, st.session_state.history, logo_path=None)
        with open(out_path, "rb") as f:
            st.download_button("Baixar PDF", data=f.read(), file_name=out_path, mime="application/pdf")
    except Exception as e:
        st.error(f"Falha ao exportar PDF: {e}")
