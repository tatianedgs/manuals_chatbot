# app_streamlit.py — NUPETR/IDEMA (design verde) + coleção automática por modo/dim
from __future__ import annotations

# ── estabilidade no Streamlit Cloud (desliga watcher) ───────────────────────────
import os as _os
_os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
_os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# ── imports ─────────────────────────────────────────────────────────────────────
import os
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from src.settings import SETTINGS
from src.rag import ingest_pdfs, retrieve_top_k
from src.milvus_utils import connect, drop_collection  # usados só no modo admin

# exportar conversa (se existir no projeto)
try:
    from src.export_pdf import export_chat_pdf
    _EXPORT_OK = True
except Exception:
    _EXPORT_OK = False

# backends
try:
    from src.llm_router import EmbeddingsCloud, EmbeddingsLocal, LLMCloud, LLMLocal
    _LOCAL_OK = True
except Exception:
    from src.llm_router import EmbeddingsCloud, LLMCloud
    EmbeddingsLocal = None  # type: ignore
    LLMLocal = None         # type: ignore
    _LOCAL_OK = False


# ── helper: nomeia a coleção conforme modo + dimensão do embedding ──────────────
def collection_for(emb, base_name: str) -> str:
    try:
        mode_tag = "cloud" if isinstance(emb, EmbeddingsCloud) else "local"
    except Exception:
        mode_tag = "cloud"
    try:
        v = emb.encode(["__probe__"])
        if hasattr(v, "tolist"):
            v = v.tolist()
        dim = len(v[0])
    except Exception:
        dim = 0
    return f"{base_name}_{mode_tag}_{dim}d"


# ── bootstrap ───────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(
    page_title="Pareceres Técnicos — NUPETR/IDEMA-RN",
    page_icon="🧰",
    layout="wide",
)

# ── CSS (paleta verde, cartões limpos) ──────────────────────────────────────────
st.markdown("""
<style>
:root{
  --brand:#12806a;    /* verde principal */
  --brand-2:#0e6a57;
  --bg:#f3f7f5;       /* fundo claro com leve verde */
  --card:#ffffff;
  --border:#e6ece9;
  --text:#0e1512;
  --muted:#5f6b68;
}
.stApp{ background:var(--bg); }

/* Sidebar com gradiente verde suave */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #e7f5ef 0%, #f3f7f5 35%, #f3f7f5 100%);
  border-right:1px solid var(--border);
  padding-top: .4rem;
}

/* Cartões (conteúdo) */
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:14px;
  padding:.9rem 1rem;
  box-shadow:0 1px 2px rgba(0,0,0,.03);
}

/* Inputs/botões */
.stTextInput>div>div>input,
.stTextArea textarea,
.stSelectbox>div>div>div{
  border-radius:12px !important;
  border:1px solid var(--border) !important;
}
.stTextInput>div>div>input:focus{
  border-color:var(--brand) !important;
}
.stButton>button{
  width:100%;
  border-radius:12px;
  padding:.7rem 1rem;
  font-weight:600;
  border:1px solid var(--brand);
  color:#fff; background:var(--brand);
}
.stButton>button:hover{
  background:var(--brand-2); border-color:var(--brand-2);
}

/* Chat */
[data-testid="stChatMessage"]{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:14px;
  padding:.75rem .9rem;
  box-shadow:0 1px 2px rgba(0,0,0,.03);
}

/* Títulos pequenos na sidebar */
.sb-title{ font-weight:700; font-size:.95rem; color:var(--text); margin:.25rem 0 .4rem; }
.sb-note{ color:var(--muted); font-size:.85rem; }

/* Header principal */
.header{
  display:flex; align-items:center; gap:.75rem;
  margin-bottom:.6rem;
}
.header h1{
  font-size:1.25rem; margin:0; color:var(--text);
}
.header p{
  margin:0; color:var(--muted);
}
.badge{
  display:inline-block; background:#e9f7f3; color:var(--brand);
  border:1px solid var(--border); border-radius:999px; padding:.15rem .55rem; font-size:.8rem;
}

/* Espaçadores leves (sem barras que confundem) */
.spacer{ height:.6rem; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR (logo, PDFs, filtros, modo, chave e ações principais) ──────────────
with st.sidebar:
    # LOGO (ajuste o caminho se necessário)
    logo_ok = False
    for candidate in ("assets/logo_idema.png", "assets/logo_idema.pngjpeg", "assets/logo_idema.jpg"):
        try:
            st.image(candidate, use_container_width=True); logo_ok = True; break
        except Exception:
            pass
    if not logo_ok:
        st.write("**IDEMA/RN**")

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # PDFs
    st.markdown('<div class="sb-title">PDFs</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Selecione ou arraste arquivos (PDF)",
        type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Filtros (no sidebar)
    st.markdown('<div class="sb-title">Filtros</div>', unsafe_allow_html=True)
    tipo_lic = st.text_input("Tipo de Licença", placeholder="ex.: RLO")
    tipo_emp = st.text_input("Tipo de Empreendimento", placeholder="ex.: POÇO")
    expr = f'tipo_licenca == "{tipo_lic or ""}" && tipo_empreendimento == "{tipo_emp or ""}"'

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Modo do modelo
    st.markdown('<div class="sb-title">Modo do modelo</div>', unsafe_allow_html=True)
    if _LOCAL_OK:
        mode = st.radio(" ", ["OpenAI (com chave)", "Modelo Local (sem chave)"], index=0, label_visibility="collapsed")
    else:
        mode = st.radio(" ", ["OpenAI (com chave)"], index=0, label_visibility="collapsed")
        st.caption("Modelo Local indisponível neste ambiente.")

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Chave OpenAI + instrução
    st.markdown('<div class="sb-title">Chave da OpenAI</div>', unsafe_allow_html=True)
    show_key = st.checkbox("mostrar chave", value=False)
    key_type = "default" if show_key else "password"

    key_source = st.radio("Origem", ["Minha chave", "Chave do app (secrets)"], index=0, horizontal=True)
    if key_source == "Minha chave":
        user_key = st.text_input("Cole sua chave (sk-...)", type=key_type, label_visibility="collapsed")
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
            st.warning("Nenhuma OPENAI_API_KEY definida nos Secrets.")

    st.markdown(
        '<div class="sb-note">💡 '
        'Como obter sua chave: acesse '
        '<a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com/api-keys</a>. '
        'A chave é usada só nesta sessão e não é armazenada.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Botões principais
    if st.button("📥 Processar PDFs", use_container_width=True, disabled=uploads is None or len(uploads) == 0):
        if not uploads:
            st.warning("Envie pelo menos um PDF.")
        else:
            pairs = [(f.name, f.read()) for f in uploads]

            # backend conforme modo
            emb, llm = None, None
            if mode.startswith("OpenAI"):
                if not SETTINGS.openai_api_key:
                    st.error("Informe uma OPENAI_API_KEY.")
                else:
                    emb = EmbeddingsCloud(); llm = LLMCloud()
            else:
                emb = EmbeddingsLocal(); llm = LLMLocal()  # type: ignore

            if emb is not None:
                try:
                    coll_name = collection_for(emb, SETTINGS.milvus_collection)
                    st.session_state["coll_name"] = coll_name  # guarda p/ conversa

                    with st.spinner(f"Processando e indexando em **{coll_name}**..."):
                        n = ingest_pdfs(
                            encoder=emb,  # type: ignore
                            files=pairs,
                            tipo_licenca=tipo_lic or "—",
                            tipo_empreendimento=tipo_emp or "—",
                            collection_name=coll_name,
                        )
                    st.success(f"✅ {n} trechos inseridos em **{coll_name}**.")
                except Exception as e:
                    st.error(f"Falha na indexação: {e}")
                    st.exception(e)

    if st.button("🧹 Limpar histórico", use_container_width=True):
        st.session_state.history = []
        st.success("Histórico limpo.")

    # Admin oculto (acessar com ?admin=1)
    qp = st.experimental_get_query_params()
    if qp.get("admin", ["0"])[0] in ("1", "true", "on"):
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-title">Admin (somente equipe)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔌 Testar conexão", use_container_width=True):
                try: connect(); st.success("Conexão OK.")
                except Exception as e: st.error(f"Falha: {e}"); st.exception(e)
        with c2:
            if st.button("🗑️ Clear Collection", use_container_width=True):
                try:
                    # se tiver uma coleção ativa na sessão, limpa ela
                    drop_collection(st.session_state.get("coll_name", SETTINGS.milvus_collection))
                    st.success("Coleção removida.")
                except Exception as e:
                    st.error(f"Falha ao remover: {e}"); st.exception(e)

# ── Cabeçalho central ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header">
      <div>
        <h1>Pareceres Técnicos — NUPETR/IDEMA-RN</h1>
        <p>Assistente para consultar e elaborar pareceres a partir de trechos de PDFs internos (RAG + Milvus).</p>
      </div>
      <span class="badge">beta</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Conversa ───────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []

st.subheader("Conversa")
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

question = st.chat_input("Digite sua pergunta aqui…")
if question:
    st.session_state.history.append(("user", question))

    # backend igual ao modo da sidebar
    emb, llm = None, None
    if _LOCAL_OK and "Local" in (locals().get("mode") or ""):
        emb = EmbeddingsLocal(); llm = LLMLocal()  # type: ignore
    else:
        if not SETTINGS.openai_api_key:
            with st.chat_message("assistant"):
                st.markdown("Defina sua **OPENAI_API_KEY** na barra lateral para eu responder.")
        else:
            emb = EmbeddingsCloud(); llm = LLMCloud()

    try:
        # usa a mesma convenção de coleção (ou a última que indexou)
        coll_name = st.session_state.get("coll_name")
        if not coll_name and emb is not None:
            coll_name = collection_for(emb, SETTINGS.milvus_collection)

        hits = retrieve_top_k(
            encoder=emb,                                  # type: ignore
            query=question,
            collection_name=coll_name or SETTINGS.milvus_collection,
            top_k=5,
            expr=f'tipo_licenca == "{tipo_lic or ""}" && tipo_empreendimento == "{tipo_emp or ""}"',
        )
        ctx = [h["text"] for h in hits]
        answer = llm.answer(question, ctx) if llm else "Configure o backend na barra lateral."

        if hits:
            refs = "\n".join(
                f"• {h['fonte']} (p.{h['pagina']}) — {h['tipo_licenca']}/{h['tipo_empreendimento']}"
                for h in hits
            )
            final = f"{answer}\n\n**Fontes consultadas:**\n{refs}"
        else:
            final = answer
    except Exception as e:
        final = f"Falha ao buscar/gerar resposta: {e}"
        st.exception(e)

    with st.chat_message("assistant"):
        st.markdown(final)
    st.session_state.history.append(("assistant", final))

# ── Exportar conversa (se disponível) ───────────────────────────────────────────
if _EXPORT_OK:
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    if st.button("🧾 Exportar conversa (PDF)"):
        try:
            out = "conversa_nupetr.pdf"
            export_chat_pdf(out, st.session_state.history, logo_path=None)
            with open(out, "rb") as f:
                st.download_button("Baixar PDF", data=f.read(), file_name=out, mime="application/pdf")
        except Exception as e:
            st.error(f"Falha ao exportar PDF: {e}")
            st.exception(e)
