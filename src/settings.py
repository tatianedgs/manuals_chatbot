from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Carrega .env local se existir (não interfere no Streamlit Secrets)
load_dotenv()


@dataclass
class Settings:
    # OpenAI (opcional): pode vir dos Secrets ou do input do usuário na sidebar
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Milvus / Zilliz
    milvus_uri: str = os.getenv("MILVUS_URI", "")            # Ex.: https://...serverless....zilliz.com   (SEM :19530 em Serverless)
    milvus_user: str = os.getenv("MILVUS_USER", "")           # só para Dedicated
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "")   # só para Dedicated
    milvus_token: str = os.getenv("MILVUS_TOKEN", "")         # Serverless (API Key)
    milvus_db: str = os.getenv("MILVUS_DB", "default")        # ignorado em Serverless, mas mantemos
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "docs_nupetr")

    # (Opcional) Modo local — se você vier a usar Ollama
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


SETTINGS = Settings()
