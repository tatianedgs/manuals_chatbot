from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Carrega .env local (não atrapalha secrets do Streamlit Cloud)
load_dotenv()

# Tenta ler também do st.secrets quando estiver no Streamlit Cloud
try:
    import streamlit as st  # type: ignore
    _SECRETS = dict(st.secrets)
except Exception:
    _SECRETS = {}

def _get(name: str, default: str = "") -> str:
    """Busca primeiro em st.secrets, depois em variáveis de ambiente."""
    v = _SECRETS.get(name)
    if v is not None and str(v).strip():
        return str(v)
    v = os.getenv(name)
    if v is not None and str(v).strip():
        return str(v)
    return default

@dataclass
class Settings:
    # OpenAI
    openai_api_key: str = _get("OPENAI_API_KEY", "")

    # Zilliz/Milvus (Serverless)
    milvus_uri: str = _get("MILVUS_URI", "")         # ex.: https://in03-...cloud.zilliz.com (SEM :19530)
    milvus_token: str = _get("MILVUS_TOKEN", "")     # API Key (token) copiado em API Keys → View
    milvus_collection: str = _get("MILVUS_COLLECTION", "docs_nupetr")

    # Campos legados (não usados em Serverless; apenas p/ dedicated)
    milvus_user: str = _get("MILVUS_USER", "")
    milvus_password: str = _get("MILVUS_PASSWORD", "")
    milvus_db: str = _get("MILVUS_DB", "default")

    # (opcional) modo local
    ollama_host: str = _get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.1:8b")

SETTINGS = Settings()
