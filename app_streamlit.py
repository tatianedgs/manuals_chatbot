# app_streamlit.py (versão focada em DESIGN)
from __future__ import annotations

# ── garante estabilidade no Streamlit Cloud ─────────────────────────────────────
import os as _os
_os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
_os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# ── imports ─────────────────────────────────────────────────────────────────────
import os
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv

from src.settings import SETTINGS
from src.milvus_utils import (
    connect, drop_collection, get_or_create_collection, insert_records
)
from src.rag import ingest_pdfs, retrieve_top_k

# exportar PDF (se você tiver no projeto)
try:
    from src.export_pdf import export_chat_pdf
    _EXPORT_OK = True
except Exception:
    _EXPORT_OK = False

# backends (protege modo Local no Cloud)
try:
    from src.llm_router import EmbeddingsCloud, EmbeddingsLocal, LLMCloud, LLMLocal
    _LOCAL_OK = True
except Exception:
    from src.llm_router import EmbeddingsCloud, LLMCloud
    EmbeddingsLocal = None  # type: ignore
    LLMLocal = None         # type: ignore
    _LOCAL_OK = False


# ── bootstrap ───────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(
    page_title="Pareceres Técnicos — NUPETR",
    page_icon="🧰",
    layout="wide",
)

# ── estilização (apenas CSS/visual) ─────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* tipografia e cores */
      :root {
        --brand: #127c7e;       /* verde água IDEMA */
        --brand-2: #0f5e60;
        --bg: #f6f9fb;
        --text: #101418;
        --muted: #5f6b7a;
        --card: #ffffff;
        --border: #e6ebf2;
      }
      .stApp { background: var(--bg); }
      .main > div { padding-top: 0.25rem; }

      /* título e subtítulo */
      .app-header { 
        display:flex; align-items:center; gap:.75rem;
        padding: .75rem 1rem; background: var(--card);
        border: 1px solid var(--border); border-radius: 14px;
        box-shadow: 0 1px 2px rgba(16,20,24,.03);
        margin-bottom: .75rem;
      }
      .app-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 0; }
      .app-caption { font-size: .9rem; color: var(--muted); margin: 0; }

      /* cards */
      .card {
        background: var(--card); border: 1px solid var(--border); border-radius: 14px;
        padding: .9rem 1rem; box-shadow: 0 1px 2px rgba(16,20,24,.03);
      }

      /* inputs + botões */
      .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div>div {
        border-radius: 12px !important; border:1px solid var(--border) !important;
      }
      .stTextInput>div>div>input::placeholder { color:#9aa6b2; }
      .stTextInput>div>div>input:focus { border-color: var(--brand) !important; }
      .stButton>button {
        width: 100%; border-radius: 12px; padding:.7rem 1rem; font-weight:600;
        border: 1px solid var(--brand); color: #fff; background: var(--brand);
      }
      .stButton>button:hover { background: var(--brand-2); border-color: var(--brand-2); }
      .stDownloadButton>button { width:100%; }

      /* chat: deixa mais “clean” */
      [data-testid="stChatMessage"] {
        background: var(--card); border:1px solid var(--border); border-radius:14px;
        padding: .75rem .9rem; box-shadow: 0 1px 2px rgba(16,20,24,.03);
      }

      /* sidebar */
      section[data-testid="stSidebar"] {
        background: #0f0f0f00;
      }
      .sb-card {
        background: var(--card); border:1px solid var(--border); border-radius:14px;
        padding:.8rem; box-shadow: 0 1px 2px rgba(16,20,24,.03);
        margin-bottom:.75rem;
      }
      .sb-title { font-weight: 700; font-size: .95rem; color: var(--text); margin-bottom:.5rem; }
      .sb-muted { color: var(--muted); font-size:.85rem; }
      .sb-btn > button { width:100%; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR (logo + controles) ─────────────────────────────────────────────────
with st.sidebar:
    # logo
    logo_ok = False
    for candidate in ("assets/logo_idema.pngjpeg", "assets/logo_idema.png", "assets/logo_idema.jpg"):
        try:
            st.image(candidate, use_container_width=True)
            logo_ok = True
            break
        except Exception:
            pass
    if not logo_ok:
        st.markdown("**IDEMA/RN**")

    st.markdown('<div class="sb-card">', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Modo do modelo</div>', unsafe_allow_html=True)
    if _LOCAL_OK:
        mode = st.radio(" ", ["OpenAI (com chave)", "Modelo Leve (sem chave)"], index=0, label_visibility="collapsed")
    else:
        mode = st.radio(" ", ["OpenAI (com chave)"], index=0, label_visibility="collapsed")
        st.caption("Modelo Local indisponível neste ambiente.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-card">', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">OPENAI_API_KEY</div>', unsafe_allow_html=True)
    # chave do usuário vs secrets
    key_source = st.radio("Origem", ["Minha chave", "Chave do app (secrets)"], index=0, horizontal=True)
    show_key = st.checkbox("mostrar", value=False)
    key_type = "default" if show_key else "password"
    if key_source == "Minha chave":
        user_key = st.text_input("Cole sua chave", type=key_type, label_visibility="collapsed", placeholder="sk-...")
        if user_key.strip():
            SETTINGS.openai_api_key = user_key.strip()
            os.environ["OPENAI_API_KEY"] = SETTINGS.openai_api_key
    else:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
        if secret_key:
            SETTINGS.openai_api_key = secret_key
            os.environ["OPENAI_API_KEY"] = secret_key
            st.caption("Usando a chave do app (secrets).")
        else:
            st.warning("Nenhuma OPENAI_API_KEY em Secrets.")
    st.markdown(
        '<span class="sb-muted">💡 <a href="https://platform.openai.com/api-keys" target="_blank">Como obter sua OpenAI API key</a></span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # PDFs (sidebar enxuta, botão grande embaixo)
    st.markdown('<div class="sb-card">', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">PDFs</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Arraste/solte ou navegue", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # Admin (recolhível)
    with st.expander("⚙️ Admin"):
        st.caption("Diagnóstico rápido do backend (Milvus/Zilliz).")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔌 Testar conexão", use_container_width=True):
                try:
                    connect()
                    st.success("Conexão OK (HTTPS + TOKEN).")
                except Exception as e:
                    st.error(f"Falha na conexão: {e}")
                    st.exception(e)
        with c2:
            if st.button("🗑️ Clear Collection", use_container_width=True):
                try:
                    drop_collection(SETTINGS.milvus_collection)
                    st.success(f"Coleção '{SETTINGS.milvus_collection}' removida.")
                except Exception as e:
                    st.error(f"Falha ao remover: {e}")
                    st.exception(e)

        st.divider()
        st.caption("Teste de escrita (1 registro).")
        if st.button("✍️ Inserir registro de teste", use_container_width=True):
            try:
                # precisamos do embedder para pegar a dimensão
                tmp_emb = EmbeddingsCloud() if (key_source == "Chave do app (secrets)" or SETTINGS.openai_api_key) else None
                if tmp_emb is None:
                    st.error("Inicialize o modo do modelo (OpenAI/Local) primeiro.")
                else:
                    vec = tmp_emb.encode(["ping"])
                    if hasattr(vec, "tolist"):
                        vec = vec.tolist()
                    dim = len(vec[0])
                    col = get_or_create_collection(SETTINGS.milvus_collection, dim=dim)
                    registro = [{
                        "embedding": vec[0],
                        "text": "registro_teste",
                        "fonte": "teste",
                        "pagina": 1,
                        "tipo_licenca": "RLO",
                        "tipo_empreendimento": "POÇO",
                    }]
                    insert_records(col, registro)
                    st.success("✅ Inserção de teste concluída!")
            except Exception as e:
                st.error(f"❌ Falha no teste de escrita: {e}")
                st.exception(e)

# ── CABEÇALHO + FILTROS + BARRA DE PERGUNTA ────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
      <div>
        <p class="app-title">NUPETR/IDEMA — Chat de Parecer Técnico</p>
        <p class="app-caption">Assistente RAG para PDFs internos. As respostas sempre citam as fontes.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# filtros (duas colunas, compactos no topo)
flt = st.container()
with flt:
    c1, c2 = st.columns(2)
    with c1:
        tipo_lic = st.text_input("Tipo de Licença", placeholder="ex.: RLO")
    with c2:
        tipo_emp = st.text_input("Tipo de Empreendimento", placeholder="ex.: POÇO")
    expr = f'tipo_licenca == "{tipo_lic or ""}" && tipo_empreendimento == "{tipo_emp or ""}"'

# barra de pergunta hero
st.markdown('<div class="card">', unsafe_allow_html=True)
question_top = st.text_input(" ", placeholder="Como posso ajudar com suas dúvidas sobre pareceres?", label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

# ── MODO / EMBEDDERS ───────────────────────────────────────────────────────────
emb = None
llm = None
if _LOCAL_OK and "Modelo Leve" in (globals().get("mode") or ""):
    emb = EmbeddingsLocal()  # type: ignore
    llm = LLMLocal()         # type: ignore
else:
    # OpenAI
    if not SETTINGS.openai_api_key:
        st.warning("Informe uma OPENAI_API_KEY para usar o modo OpenAI.")
    else:
        emb = EmbeddingsCloud()
        llm = LLMCloud()

# ── AÇÕES PRINCIPAIS (Processar PDFs) ──────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
process_disable = (emb is None) or (uploads is None or len(uploads) == 0)
if st.button("📥 Processar PDFs", disabled=process_disable):
    if not uploads:
        st.warning("Envie pelo menos um PDF.")
    else:
        pairs = [(f.name, f.read()) for f in uploads]
        try:
            with st.spinner("Processando, gerando embeddings e inserindo no Milvus..."):
                n = ingest_pdfs(
                    encoder=emb,  # type: ignore
                    files=pairs,
                    tipo_licenca=tipo_lic or "—",
                    tipo_empreendimento=tipo_emp or "—",
                    collection_name=SETTINGS.milvus_collection,
                )
            st.success(f"✅ {n} trechos inseridos no Milvus.")
        except Exception as e:
            st.error(f"Falha na indexação: {e}")
            st.exception(e)
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ── CONVERSA ───────────────────────────────────────────────────────────────────
# (processa a pergunta do topo como se fosse chat)
if "history" not in st.session_state:
    st.session_state.history = []

if question_top:
    st.session_state.history.append(("user", question_top))
    try:
        hits = retrieve_top_k(
            encoder=emb,  # type: ignore
            query=question_top,
            collection_name=SETTINGS.milvus_collection,
            top_k=5,
            expr=expr,
        )
        ctx = [h["text"] for h in hits]
        answer = llm.answer(question_top, ctx) if llm else "Configure o backend para responder."
        if hits:
            refs = "\n".join(
                f"• {h['fonte']} (p.{h['pagina']}) — {h['tipo_licenca']}/{h['tipo_empreendimento']}"
                for h in hits
            )
            final = f"{answer}\n\n**Fontes consultadas:**\n{refs}"
        else:
            final = answer
        st.session_state.history.append(("assistant", final))
    except Exception as e:
        st.session_state.history.append(("assistant", f"Falha ao buscar/gerar resposta: {e}"))
        st.exception(e)

st.subheader("Conversa")
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# Exportar conversa (se existir função)
if _EXPORT_OK:
    st.divider()
    if st.button("🧾 Exportar conversa (PDF)"):
        try:
            out = "conversa_nupetr.pdf"
            export_chat_pdf(out, st.session_state.history, logo_path=None)
            with open(out, "rb") as f:
                st.download_button("Baixar PDF", data=f.read(), file_name=out, mime="application/pdf")
        except Exception as e:
            st.error(f"Falha ao exportar PDF: {e}")
            st.exception(e)
