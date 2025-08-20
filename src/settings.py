import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # Milvus / Zilliz
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_user: str = os.getenv("MILVUS_USER", "")
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "")
    milvus_token: str = os.getenv("MILVUS_TOKEN", "")  # << usamos no Zilliz Cloud
    milvus_db: str = os.getenv("MILVUS_DB", "nupe")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "docs_nupetr")
    # LLM local (opcional)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

SETTINGS = Settings()
