from dataclasses import dataclass
import os

def _get(name: str, default: str = "") -> str:
    # Streamlit Cloud injeta os secrets como variáveis de ambiente
    return os.getenv(name, default)

@dataclass
class _Settings:
    openai_api_key: str = _get("OPENAI_API_KEY", "")
    milvus_uri: str = _get("MILVUS_URI", "https://localhost:19530")
    milvus_user: str = _get("MILVUS_USER", "")
    milvus_password: str = _get("MILVUS_PASSWORD", "")
    milvus_db: str = _get("MILVUS_DB", "nupe")
    milvus_collection: str = _get("MILVUS_COLLECTION", "docs_nupetr")
    # Opcional (modo local/ollama)
    ollama_host: str = _get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _get("OLLAMA_MODEL", "llama3.1:8b")

SETTINGS = _Settings()
