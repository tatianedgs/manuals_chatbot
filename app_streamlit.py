# app_streamlit.py — NUPETR/IDEMA (design verde) + EXTRATIVA (sem LLM)
# ordem do chat corrigida + citações limpas + gate por senha + descrição do projeto (no sidebar)
from __future__ import annotations

# estabilidade no Streamlit Cloud
import os as _os
_os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
_os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import os
from typing import List, Tuple, Sequence, Dict
import streamlit as st
from dotenv import load_dotenv
from collections import OrderedDict
from pathlib import Path
import glob

from src.settings import SETTINGS
from src.rag import ingest_pdfs, retrieve_top_k
from src.llm_router import EmbeddingsCloud, EmbeddingsLocal, LLMCloud, LiteLocal

# exportar conversa (se existir)
try:
    from src.export_pdf import export_chat_pdf
    _EXPORT_OK = True
except Exception:
    _EXPORT_OK = False


# ---------- helpers visuais e de formatação ----------
def _try_show_logo():
    """Mostra a logo do IDEMA (ordem: secrets → assets/ glob → fallback texto)."""
    logo_hint = st.secrets.get("LOGO_PATH", "")
    candidates: List[str] = []
    if logo_hint:
        candidates.append(logo_hint)

    assets = Path("assets")
    if assets.exists():
        for ext in ("png", "jpg", "jpeg"):
            candidates.extend(glob.glob(str(assets / f"*idema*.{ext}")))
            candidates.extend(glob.glob(str(assets / f"*IDEMA*.{ext}")))

    candidates.extend([
        "assets/logo_idema.png",
        "assets/logo_idema.jpg",
        "assets/logo_idema.jpeg",
    ])

    shown = False
    for cand in candidates:
        try:
            st.image(cand, use_container_width=True)
            shown = True
            break
        except Exception:
            continue
    if not shown:
        st.write("**IDEMA/RN**")

def collection_for(emb, base_name: str) -> str:
    """Nomeia coleção conforme modo + dimensão dos embeddings (evita 3072×384)."""
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

def format_citations(hits: Sequence[Dict]) -> str:
    """
    Agrupa por documento e lista páginas únicas, ordenadas.
    """
    groups = OrderedDict()
    for h in hits:
        key = (h.get("fonte") or "", h.get("tipo_licenca") or "", h.get("tipo_empreendimento") or "")
        if key not in groups:
            groups[key] = set()
        try:
            groups[key].add(int(h.get("pagina", 0)))
        except Exception:
            pass

    if not groups:
        return ""

    lines = []
    for (fonte, tlic, temp), pages in groups.items():
        fname = os.path.basename(fonte) if fonte else "—"
        pages_sorted = sorted([p for p in pages if p > 0])
        if pages_sorted:
            pages_txt = ", ".join(str(p) for p in pages_sorted)
            lines.append(f"• {fname} — p. {pages_txt} ({tlic}/{temp})")
        else:
            lines.append(f"• {fname} ({tlic}/{temp})")

    return "\n".join(lines)


# ---------- bootstrap ----------
load_dotenv()
st.set_page_config(page_title="Pareceres Técnicos — NUPETR/IDEMA-RN", page_icon="🧰", layout="wide")

# ---------- GATE por senha (APP_PASSCODE nos Secrets) ----------
REQUIRED_CODE = st.secrets.get("APP_PASSCODE", "")
if REQUIRED_CODE:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        st.title("🔒 Acesso restrito — NUPETR/IDEMA-RN")
        st.caption("Informe o código de acesso fornecido pela equipe do NUPETR.")
        code = st.text_input("Código de acesso", type="password")
        if st.button("Entrar"):
            if code == REQUIRED_CODE:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Código incorreto.")
        st.stop()

# ---------- CSS (paleta verde) ----------
st.markdown("""
<style>
:root{ --brand:#12806a; --brand-2:#0e6a57; --bg:#f3f7f5; --card:#fff; --border:#e6ece9; --text:#0e1512; --muted:#5f6b68; }
.stApp{ background:var(--bg); }
section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#e7f5ef 0%,#f3f7f5 35%,#f3f7f5 100%); border-right:1px solid var(--border); padding-top:.4rem; }
.card{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:.9rem 1rem; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div>div{ border-radius:12px!important; border:1px solid var(--border)!important; }
.stTextInput>div>div>input:focus{ border-color:var(--brand)!important; }
.stButton>button{ width:100%; border-radius:12px; padding:.7rem 1rem; font-weight:600; border:1px solid var(--brand); color:#fff; background:var(--brand); }
.stButton>button:hover{ background:var(--brand-2); border-color:var(--brand-2); }
[data-testid="stChatMessage"]{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:.75rem .9rem; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.sb-title{ font-weight:700; font-size:.95rem; color:var(--text); margin:.25rem 0 .4rem; }
.sb-note{ color:var(--muted); font-size:.85rem; }
.header{ display:flex; align-items:center; gap:.75rem; margin-bottom:.6rem; }
.header h1{ font-size:1.25rem; margin:0; color:var(--text); }
.header p{ margin:0; color:var(--muted); }
.badge{ display:inline-block; background:#e9f7f3; color:var(--brand); border:1px solid var(--border); border-radius:999px; padding:.15rem .55rem; font-size:.8rem; }
.spacer{ height:.6rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    _try_show_logo()

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # PDFs
    st.markdown('<div class="sb-title">PDFs</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Selecione ou arraste arquivos (PDF)", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Filtros
    st.markdown('<div class="sb-title">Filtros</div>', unsafe_allow_html=True)
    tipo_lic = st.text_input("Tipo de Licença", placeholder="ex.: RLO")
    tipo_emp = st.text_input("Tipo de Empreendimento", placeholder="ex.: POÇO")

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Modo
    st.markdown('<div class="sb-title">Modo</div>', unsafe_allow_html=True)
    mode = st.radio(" ", ["OpenAI (com chave)", "Extrativa (sem LLM)"], index=0, label_visibility="collapsed")

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Chave da OpenAI (opcional) — com explicação e quem paga
    st.markdown('<div class="sb-title">Chave da OpenAI (opcional)</div>', unsafe_allow_html=True)
    use_secrets = st.radio("Quem vai fornecer a chave?", ["Usar a chave do app (usuário Tatiane)", "Usar minha própria chave"], index=0)
    key_value = ""

    if "própria" in use_secrets.lower():
        key_value = st.text_input("Cole sua chave (sk-...)", type="password", label_visibility="collapsed")
        if key_value:
            os.environ["OPENAI_API_KEY"] = key_value
            SETTINGS.openai_api_key = key_value
            st.caption("💳 **Custo**: debitado na **sua** conta OpenAI.")
        else:
            st.caption("Cole sua chave acima para usar modelos da OpenAI.")
    else:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
        if secret_key:
            os.environ["OPENAI_API_KEY"] = secret_key
            SETTINGS.openai_api_key = secret_key
            st.caption("💳 **Custo**: Usar a **chave do app** gera um custo (usuário Tatiane).")
        else:
            st.warning("Nenhuma chave do app definida em Secrets. Cole a sua acima ou peça à TI.")

    with st.expander("Como obter a chave?"):
        st.markdown(
            "- Acesse **platform.openai.com/api-keys** (faça login).\n"
            "- Clique em **Create new secret key** e copie o código `sk-...`.\n"
            "- **Cole** no campo acima. A chave é usada **somente nesta sessão** e **não é armazenada**."
        )

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Botões
    if st.button("📥 Processar PDFs", use_container_width=True, disabled=uploads is None or len(uploads) == 0):
        if not uploads:
            st.warning("Envie pelo menos um PDF.")
        else:
            pairs = [(f.name, f.read()) for f in uploads]

            # Backend: embeddings para indexar
            if mode.startswith("OpenAI"):
                if not SETTINGS.openai_api_key:
                    st.error("Forneça uma OPENAI_API_KEY ou selecione o modo 'Extrativa (sem LLM)'.")
                    emb = None
                else:
                    emb = EmbeddingsCloud()
            else:  # Extrativa
                emb = EmbeddingsCloud() if SETTINGS.openai_api_key else EmbeddingsLocal()

            if emb is not None:
                try:
                    coll_name = collection_for(emb, SETTINGS.milvus_collection)
                    st.session_state["coll_name"] = coll_name

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

    # ---- Sobre o projeto (AGORA NO SIDEBAR) ----
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    with st.expander("ℹ️ Sobre este projeto"):
        st.markdown(
            """
**NUPETR — Núcleo de Atividades Petrolíferas do IDEMA/RN**  
Assistente para apoiar analistas na elaboração de **pareceres técnicos** e **análises de estudos**, a partir de trechos dos documentos internos (RAG com Milvus/Zilliz).  
**Primeiro módulo:** *RLO_POÇO*.

**Desenvolvimento:** Tatiane Gois e **Sinara Carla** (NUPETR/IDEMA-RN).  
**Tecnologias:** Streamlit, Python, pypdf, Milvus/Zilliz, embeddings OpenAI (ou locais) e **modo extrativo** sem LLM.  
As respostas são **ancoradas** nos trechos indexados e exibem *Fontes consultadas*.
            """
        )

# ---------- Header ----------
st.markdown("""
<div class="header">
  <div>
    <h1>Pareceres Técnicos — NUPETR/IDEMA-RN</h1>
    <p>Consulta e elaboração de pareceres a partir de trechos dos PDFs internos (RAG + Milvus).</p>
  </div>
  <span class="badge">beta</span>
</div>
""", unsafe_allow_html=True)

# ---------- Conversa (ordem corrigida) ----------
if "history" not in st.session_state:
    st.session_state.history: List[Tuple[str, str]] = []

st.subheader("Conversa")

# render histórico já existente
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# input do usuário
question = st.chat_input("Digite sua pergunta aqui…")

if question:
    # 1) mostra a MENSAGEM DO USUÁRIO imediatamente
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    # 2) prepara backend conforme modo
    if mode.startswith("OpenAI"):
        if not SETTINGS.openai_api_key:
            with st.chat_message("assistant"):
                st.markdown("Defina sua **OPENAI_API_KEY** na barra lateral ou mude para **Extrativa (sem LLM)**.")
            st.stop()
        emb = EmbeddingsCloud()
        answerer = LLMCloud()
    else:
        emb = EmbeddingsCloud() if SETTINGS.openai_api_key else EmbeddingsLocal()
        answerer = LiteLocal()

    # 3) placeholder da resposta (mostra 'pensando...' enquanto busca)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_pensando…_")

        try:
            coll_name = st.session_state.get("coll_name")
            if not coll_name:
                coll_name = collection_for(emb, SETTINGS.milvus_collection)

            hits = retrieve_top_k(
                encoder=emb,  # type: ignore
                query=question,
                collection_name=coll_name,
                top_k=5,
                expr=f'tipo_licenca == "{tipo_lic or ""}" && tipo_empreendimento == "{tipo_emp or ""}"',
            )
            ctx = [h["text"] for h in hits]
            answer_text = answerer.answer(question, ctx)  # type: ignore

            refs_block = format_citations(hits)
            if refs_block:
                final = f"{answer_text}\n\n**Fontes consultadas:**\n{refs_block}"
            else:
                final = answer_text
        except Exception as e:
            final = f"Falha ao buscar/gerar resposta: {e}"
            st.exception(e)

        # 4) substitui o placeholder pela resposta final
        placeholder.markdown(final)

    # 5) salva a resposta no histórico
    st.session_state.history.append(("assistant", final))

# ---------- Exportar ----------
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
