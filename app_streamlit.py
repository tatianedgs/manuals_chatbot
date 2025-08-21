from __future__ import annotations

import os
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv

from src.settings import SETTINGS
from src.milvus_utils import drop_collection, connect
from src.rag import ingest_pdfs, retrieve_top_k
from src.export_pdf import export_chat_pdf

# Backends
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
st.caption("Responde com base em trechos de PDFs internos. Escolha os filtros, envie PDFs e selecione o backend.")

if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []

# ===== Sidebar: chave OpenAI =====
st.sidebar.header("🔐 Minha chave vs Chave do app")
key_mode = st.sidebar.radio("Origem da chave OpenAI", ["Minha chave", "Chave do app (secrets)"], index=0)

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
        st.sidebar.success("Usando a chave do app (secrets).")
    else:
        st.sidebar.warning("OPENAI_API_KEY não definida nos Secrets.")

# ===== Sidebar: modo =====
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
        st.sidebar.error("Informe uma OPENAI_API_KEY (cole acima ou configure em Secrets).")
    else:
        emb = EmbeddingsCloud()
        llm = LLMCloud()
else:
    emb = EmbeddingsLocal()  # type: ignore
    llm = LLMLocal()        # type: ignore

# ===== Sidebar: debug de conexão Milvus =====
st.sidebar.subheader("🧩 Milvus/Zilliz (Serverless)")
st.sidebar.caption(f"URI em uso: {SETTINGS.milvus_uri or '—'}")
st.sidebar.caption("Token carregado: " + ("✅" if len(SETTINGS.milvus_token) > 10 else "❌"))

if st.sidebar.button("🔎 Testar conexão Milvus"):
    try:
        connect()
        st.sidebar.success("Conexão OK! (URI HTTPS + TOKEN)")
    except Exception as e:
        st.sidebar.error(f"Falha na conexão: {e}")

# ===== Filtros =====
st.sidebar.subheader("Filtros por metadados")
tipo_lic = st.sidebar.selectbox("Tipo de Licença", ["RLO", "LP", "LI", "LO", "OUTROS"], index=0)
tipo_emp = st.sidebar.selectbox("Tipo de Empreendimento", ["POÇO", "ESTAÇÃO", "OLEODUTO", "BASE", "OUTROS"], index=0)
expr = f'tipo_licenca == "{tipo_lic}" && tipo_empreendimento == "{tipo_emp}"'

# ===== PDFs & Ações =====
st.sidebar.subheader("📄 PDFs")
uploads = st.sidebar.file_uploader(
    "Selecione (padrão: tipoLicenca_tipoEmpreendimento.pdf)",
    type=["pdf"], accept_multiple_files=True
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

# ===== Ingestão =====
st.subheader("📥 Indexação de PDFs")
if st.button("📌 Indexar PDFs no Milvus", disabled=(emb is None) or not uploads):
    if not uploads:
        st.warning("Envie pelo menos um PDF.")
    else:
        pairs = [(f.name, f.read()) for f in uploads]
        try:
            with st.spinner("Gerando embeddings e inserindo no Milvus..."):
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

st.divider()

# ===== Conversa =====
st.subheader("💬 Conversa")
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Digite sua pergunta aqui...")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        hits = retrieve_top_k(
            encoder=emb,  # type: ignore
            query=prompt,
            collection_name=SETTINGS.milvus_collection,
            top_k=5,
            expr=expr,
        )
        ctx = [h["text"] for h in hits]
        answer = llm.answer(prompt, ctx)  # type: ignore
        if hits:
            refs = "\n".join(f"• {h['fonte']} (p.{h['pagina']}) — {h['tipo_licenca']}/{h['tipo_empreendimento']}" for h in hits)
            final = f"{answer}\n\n**Fontes consultadas:**\n{refs}"
        else:
            final = answer
    except Exception as e:
        final = f"Falha ao buscar/gerar resposta: {e}"

    with st.chat_message("assistant"):
        st.markdown(final)
    st.session_state.history.append(("assistant", final))

st.divider()
if st.button("🧾 Exportar conversa (PDF)"):
    try:
        out = "conversa_nupetr.pdf"
        export_chat_pdf(out, st.session_state.history, logo_path=None)
        with open(out, "rb") as f:
            st.download_button("Baixar PDF", data=f.read(), file_name=out, mime="application/pdf")
    except Exception as e:
        st.error(f"Falha ao exportar PDF: {e}")
