from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Carrega um .env local se existir (não atrapalha Secrets do Streamlit)
load_dotenv()

# Tenta ler também do st.secrets (quando rodando no Streamlit Cloud)
try:
    import streamlit as st  # type: ignore
    _SECRETS = dict(st.secrets)
except Exception:
    _SECRETS = {}

def _get(name: str, default: str = "") -> str:
    """Busca primeiro em st.secrets, depois em variáveis de ambiente."""
    val = _SECRETS.get(name, None)
    if val is not None and str(val).strip() != "":
        return str(val)
    val = os.getenv(name, None)
    if val is not None and str(val).strip() != "":
        return str(val)
    return default

@dataclass
class Settings:
    # OpenAI
    openai_api_key: str = _get("OPENAI_API_KEY", "")

    # Milvus / Zilliz
    # Serverless: use o Public Endpoint HTTPS **sem** :19530
    milvus_uri: str = _get("MILVUS_URI", "")
    # Dedicated (não usado em Serverless):
    milvus_user: str = _get("MILVUS_USER", "")
    milvus_password: str = _get("MILVUS_PASSWORD", "")
    # Serverless (API Key / Token):
    milvus_token: str = _get("MILVUS_TOKEN", "")
    milvus_db: str = _get("MILVUS_DB", "default")          # ignorado em Serverless
    milvus_collection: str = _get("MILVUS_COLLECTION", "docs_nupetr")

    # (opcional) modo local
    ollama_host: str = _get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.1:8b")

SETTINGS = Settings()
