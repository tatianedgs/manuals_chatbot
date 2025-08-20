import os
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv

from src.settings import SETTINGS
from src.milvus_utils import drop_collection
from src.rag import ingest_pdfs, retrieve_top_k
from src.export_pdf import export_chat_pdf
from src.llm_router import EmbeddingsCloud, EmbeddingsLocal, LLMCloud, LLMLocal

load_dotenv()

st.set_page_config(
    page_title="NUPETR/IDEMA-RN • Chat de Parecer Técnico",
    page_icon="💼",
    layout="wide",
)

st.title("💼 NUPETR/IDEMA-RN — Chat de Parecer Técnico (RAG + Milvus)")
st.caption("As respostas citam trechos do PDF. Valide sempre as informações.")

# ----------------- Estado -----------------
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# ----------------- Sidebar -----------------
with st.sidebar:
    st.subheader("🔐 Minha chave")
    key_mode = st.radio(
        "Escolha a chave",
        ["Minha chave (formato sk...)", "Chave do app"],
        index=1,
        help="Use sua própria chave (não é salva) ou a chave configurada nas secrets do app.",
    )
    if key_mode.startswith("Minha"):
        st.session_state.user_api_key = st.text_input(
            "Cole sua chave (formato sk_...)",
            type="password",
            placeholder="sk-...",
            help="Só é usada nesta sessão.",
        )
        current_api_key = st.session_state.user_api_key.strip()
    else:
        current_api_key = os.getenv("OPENAI_API_KEY", SETTINGS.openai_api_key)

    st.markdown("---")
    st.subheader("⚙️ Modo do modelo")
    mode = st.radio(
        "Escolha o modo",
        ["OpenAI (com chave)", "Modelo Local (sem chave)"],
        index=0,
        help="No Streamlit Cloud, o modo Local geralmente não está disponível.",
    )

    st.markdown("---")
    st.caption("🧹 Limpar tudo")
    if st.button("Resetar coleção (Milvus)"):
        drop_collection(SETTINGS.milvus_collection)
        st.success("Coleção resetada (Milvus).")
    if st.button("Limpar conversa"):
        st.session_state.history = []
        st.experimental_rerun()

# ----------------- Filtros -----------------
col1, col2 = st.columns(2)
with col1:
    tipo_licenca = st.text_input("Tipo de Licença", placeholder="ex.: RLO")
with col2:
    tipo_emp = st.text_input("Tipo de Empreendimento", placeholder="ex.: POÇO")

# ----------------- Upload -----------------
uploaded = st.file_uploader(
    "Arraste e solte PDFs aqui",
    type=["pdf"],
    accept_multiple_files=True,
    help="Limite 200MB por arquivo.",
)

def _uploaded_to_tuples(files) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    for f in files or []:
        out.append((f.name, f.read()))
    return out

# ----------------- Botão de indexação -----------------
if st.button("📥 Indexar PDFs no Milvus", use_container_width=True):
    if not uploaded:
        st.error("Envie pelo menos um PDF.")
    else:
        # Escolha de embeddings
        if mode.startswith("OpenAI"):
            if not current_api_key:
                st.error("Forneça a chave OpenAI (401).")
            else:
                emb = EmbeddingsCloud(api_key=current_api_key)
        else:
            # Modo local (apenas se o servidor tiver o modelo local)
            try:
                emb = EmbeddingsLocal()
            except Exception as e:
                st.error(f"Modo local indisponível neste ambiente: {e}")
                emb = None

        if emb is not None:
            with st.spinner("Quebrando PDFs, gerando embeddings e gravando no Milvus..."):
                try:
                    n_chunks = ingest_pdfs(
                        encoder=emb,
                        files=_uploaded_to_tuples(uploaded),
                        tipo_licenca=tipo_licenca or "",
                        tipo_empreendimento=tipo_emp or "",
                        collection_name=SETTINGS.milvus_collection,
                    )
                    st.success(f"Indexados {n_chunks} trechos.")
                except Exception as e:
                    st.error(f"Falha na indexação: {e}")

st.markdown("---")

# ----------------- Conversa -----------------
# mensagens anteriores
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

user_q = st.chat_input("Digite sua pergunta...")
if user_q:
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.history.append(("user", user_q))

    # Seleção de LLM + embeddings para busca
    if mode.startswith("OpenAI"):
        if not current_api_key:
            st.error("Forneça a chave OpenAI (401).")
            st.stop()
        emb = EmbeddingsCloud(api_key=current_api_key)
        llm = LLMCloud(api_key=current_api_key)
    else:
        try:
            emb = EmbeddingsLocal()
            llm = LLMLocal()
        except Exception as e:
            st.error(f"Modo local indisponível neste ambiente: {e}")
            st.stop()

    # Busca semântica
    try:
        expr = None
        if tipo_licenca:
            expr = f'tipo_licenca == "{tipo_licenca}"'
        if tipo_emp:
            expr = (expr + " && " if expr else "") + f'tipo_empreendimento == "{tipo_emp}"'

        with st.spinner("Buscando trechos relevantes..."):
            hits = retrieve_top_k(
                encoder=emb,
                query=user_q,
                collection_name=SETTINGS.milvus_collection,
                top_k=5,
                expr=expr,
            )
    except Exception as e:
        st.error(f"Erro na busca: {e}")
        hits = []

    # Monta contexto para o LLM
    context = "\n\n".join([h["text"] for h in hits]) if hits else ""
    try:
        with st.spinner("Gerando resposta..."):
            answer = llm.answer(user_q, context)
    except Exception as e:
        if "401" in str(e):
            st.error("Erro 401: verifique sua chave OpenAI.")
        else:
            st.error(f"Falha ao chamar o LLM: {e}")
        answer = ""

    refs = "\n".join([
        f"• {h['fonte']} (p.{h['pagina']}) — {h['tipo_licenca']}/{h['tipo_empreendimento']}"
        for h in (hits or [])
    ])

    final = answer if not hits else f"{answer}\n\n**Fontes consultadas:**\n{refs}"

    with st.chat_message("assistant"):
        st.markdown(final)
    st.session_state.history.append(("assistant", final))

# ----------------- Exportar conversa -----------------
with st.expander("Exportar conversa"):
    if st.button("Gerar PDF"):
        try:
            path = export_chat_pdf(st.session_state.history)
            st.success(f"PDF gerado: {path}")
            st.download_button("Baixar PDF", data=open(path, "rb").read(), file_name="conversa.pdf")
        except Exception as e:
            st.error(f"Falha ao gerar PDF: {e}")
