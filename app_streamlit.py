import os
from typing import List, Tuple, Iterable

import streamlit as st
from dotenv import load_dotenv

# ===== Carrega variáveis de ambiente (.env local) =====
load_dotenv()

# ===== Imports do projeto =====
from src.settings import SETTINGS
from src.rag import ingest_pdfs, retrieve_top_k
from src.milvus_utils import drop_collection
from src.export_pdf import export_chat_pdf
from src.llm_router import EmbeddingsCloud, LLMCloud, LLMLocal

# Embeddings locais podem não estar disponíveis no Cloud
try:
    from src.llm_router import EmbeddingsLocal
    HAS_LOCAL = True
except Exception:
    EmbeddingsLocal = None  # type: ignore
    HAS_LOCAL = False

# ==========================
# Funções auxiliares (UI)
# ==========================

def setup_openai_key() -> bool:
    """Define a chave da OpenAI para esta sessão.
    Prioridade:
      1) chave que a pessoa colou na sidebar (Minha chave)
      2) chave do app (st.secrets) se existir
      3) variável de ambiente (ex.: .env local)
    Retorna True se houver chave válida configurada.
    """
    # Fallback/default (env ou secrets do deploy)
    default_key = os.getenv("OPENAI_API_KEY", "")
    if not default_key and "OPENAI_API_KEY" in st.secrets:
        default_key = st.secrets["OPENAI_API_KEY"]

    st.sidebar.markdown("### 🔐 OpenAI API key")
    fonte = st.sidebar.radio(
        "Fonte da chave",
        ["Minha chave", "Chave do app"],
        horizontal=True,
        help=(
            "Escolha 'Minha chave' para colar sua própria API key (não será salva).\n"
            "'Chave do app' usa a chave do deploy, se o mantenedor configurou no Secrets."
        ),
    )

    if fonte == "Minha chave":
        user_key = st.sidebar.text_input(
            "Cole sua chave (formato sk-...)",
            type="password",
            placeholder="sk-...",
        ).strip()
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            SETTINGS.openai_api_key = user_key
            st.sidebar.caption("✅ Usando a sua chave **apenas nesta sessão**.")
        else:
            # evita usar uma chave antiga por engano
            os.environ.pop("OPENAI_API_KEY", None)
            SETTINGS.openai_api_key = ""
    else:  # Chave do app
        if default_key:
            os.environ["OPENAI_API_KEY"] = default_key
            SETTINGS.openai_api_key = default_key
            st.sidebar.caption("🔒 Usando a chave do app (Secrets/Env).")
        else:
            st.sidebar.error(
                "Nenhuma chave padrão configurada no app. Selecione 'Minha chave' e cole a sua."
            )
            os.environ.pop("OPENAI_API_KEY", None)
            SETTINGS.openai_api_key = ""

    return bool(os.getenv("OPENAI_API_KEY", ""))


def expr_filters(tipo_licenca: str, tipo_emp: str) -> str | None:
    parts: List[str] = []
    if tipo_licenca:
        parts.append(f"tipo_licenca == \"{tipo_licenca}\"")
    if tipo_emp:
        parts.append(f"tipo_empreendimento == \"{tipo_emp}\"")
    return " and ".join(parts) if parts else None


# ==========================
# Configuração da página
# ==========================

st.set_page_config(
    page_title="NUPETR/IDEMA-RN • Chat de Parecer Técnico",
    page_icon="💼",
    layout="wide",
)

st.title("💼 NUPETR/IDEMA-RN — Chat de Parecer Técnico (RAG + Milvus)")
st.caption(
    "Assistente ancorado em manuais internos (PDF). As respostas citam os trechos fonte. "
    "Preencha os filtros (Tipo de Licença/Empreendimento), envie PDFs e escolha o backend (Nuvem ou Local)."
)

# ==========================
# Sidebar — Upload, filtros, chave, utilitários
# ==========================

# 1) Seção da chave
key_ok = setup_openai_key()

# 2) Filtros de metadados
st.sidebar.markdown("### Filtros")
tipo_licenca = st.sidebar.text_input("Tipo de Licença", value="", placeholder="ex.: RLO")
tipo_emp = st.sidebar.text_input("Tipo de Empreendimento", value="", placeholder="ex.: POÇO")

# 3) Upload de PDFs (multi)
st.sidebar.markdown("### 📄 PDFs")
uploads = st.sidebar.file_uploader(
    "Selecione (padrão: tipoLicenca_tipoEmpreendimento.pdf)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Pode arrastar vários PDFs. Limite ~200 MB por arquivo.",
)

# 4) Modo do modelo
st.sidebar.markdown("### 🧠 Modo do modelo")
opts = ["OpenAI (com chave)"]
if HAS_LOCAL:
    opts.append("Modelo Local (sem chave)")
modo = st.sidebar.radio("Escolha o modo", opts, index=0)

# 5) Ações administrativas
with st.sidebar.expander("⚙️ Admin", expanded=False):
    if st.button("🔁 Apagar coleção Milvus (recomeçar)", use_container_width=True):
        try:
            drop_collection(SETTINGS.milvus_collection)
            st.success("Coleção apagada.")
        except Exception as e:
            st.error(f"Falha ao apagar coleção: {e}")

with st.sidebar.expander("ℹ️ Dicas", expanded=False):
    st.write("- A chave colada não é salva; fica apenas na memória da sessão.")
    st.write("- Para produção, configure o Milvus/Zilliz via Secrets do deploy.")

# ==========================
# Área principal — indexação e chat
# ==========================

col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Indexação de PDFs")
    st.write(
        "Os arquivos são quebrados em trechos, vetorizados e gravados no Milvus com seus metadados."
    )

with col_b:
    st.subheader("Conversa")

# Estado da conversa
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []

# ===== Botão de indexação =====
if st.button("📥 Indexar PDFs no Milvus", type="primary", use_container_width=True):
    if not uploads:
        st.warning("Envie ao menos um PDF na barra lateral.")
    else:
        if modo.startswith("OpenAI"):
            if not key_ok:
                st.warning("Informe uma OpenAI API key na barra lateral para usar o modo Nuvem.")
                st.stop()
            emb = EmbeddingsCloud(api_key=st.session_state.get("openai_key"))  # OpenAI embeddings
        else:
            if not HAS_LOCAL:
                st.error("Modo local indisponível neste deploy.")
                st.stop()
            emb = EmbeddingsLocal()  # sentence-transformers (CPU)

        # prepara os bytes para ingestão
        files_payload: Iterable[Tuple[str, bytes]] = [
            (f.name, f.read()) for f in uploads
        ]
        try:
            n = ingest_pdfs(
                encoder=emb,
                files=files_payload,
                tipo_licenca=tipo_licenca.strip(),
                tipo_empreendimento=tipo_emp.strip(),
                collection_name=SETTINGS.milvus_collection,
            )
            st.success(f"Indexação concluída: {n} trechos inseridos.")
        except Exception as e:
            st.error(f"Falha na indexação: {e}")

# ===== Chat UI =====
for who, msg in st.session_state.history:
    with st.chat_message(who):
        st.markdown(msg)

user_q = st.chat_input("Digite sua pergunta aqui…")
if user_q:
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.history.append(("user", user_q))

    # Seletor do LLM/Embeddings para consulta
    if modo.startswith("OpenAI"):
        if not key_ok:
            st.warning("Informe uma OpenAI API key na barra lateral para usar o modo Nuvem.")
            st.stop()
        emb = EmbeddingsCloud(api_key=st.session_state.get("openai_key"))
        llm = LLMCloud(api_key=st.session_state.get("openai_key"))
    else:
        if not HAS_LOCAL:
            with st.chat_message("assistant"):
                st.markdown("[LLM local indisponível] Configure Ollama no servidor.")
            st.stop()
        emb = EmbeddingsLocal()
        llm = LLMLocal()

    # Recuperação no vetor store
    hits: List[dict] = []
    try:
        expr = expr_filters(tipo_licenca.strip(), tipo_emp.strip())
        hits = retrieve_top_k(
            encoder=emb,
            query=user_q,
            collection_name=SETTINGS.milvus_collection,
            top_k=5,
            expr=expr,
        )
    except Exception as e:
        st.warning(f"Busca no Milvus falhou: {e}")

    # Contexto para o LLM
    ctx = hits if hits else []
    try:
        answer = llm.answer(user_q, ctx)
    except Exception as e:
        answer = f"[Falha ao chamar o LLM] {e}"

    # Monta referências
    refs = "\n".join(
        [
            f"• {h['fonte']} (p.{h['pagina']}) — {h['tipo_licenca']}/{h['tipo_empreendimento']}"
            for h in hits
        ]
    )

    final_answer = f"{answer}\n\n**Fontes consultadas:**\n{refs}" if hits else answer

    with st.chat_message("assistant"):
        st.markdown(final_answer)
    st.session_state.history.append(("assistant", final_answer))

# ===== Exportar conversa em PDF =====
st.divider()
left, right = st.columns([1, 1])
with left:
    if st.button("🧹 Limpar histórico", use_container_width=True):
        st.session_state.history = []
        st.experimental_rerun()
with right:
    if st.button("🖨️ Exportar conversa (PDF)", use_container_width=True):
        try:
            pdf_bytes = export_chat_pdf(st.session_state.history)
            st.download_button(
                label="Baixar PDF",
                data=pdf_bytes,
                file_name="conversa_pareceres.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Falha ao exportar PDF: {e}")
