from typing import List, Optional
from dataclasses import dataclass
import os
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from .settings import SETTINGS

# ---------- Embeddings (OpenAI) ----------
@dataclass
class EmbeddingsCloud:
    model: str = "text-embedding-3-large"
    api_key: Optional[str] = None

    def __post_init__(self):
        key = self.api_key or SETTINGS.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY ausente. Cole sua chave na sidebar ou configure o Secrets.")
        self.client = OpenAI(api_key=key)

    def encode(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.asarray(vecs, dtype="float32")

# ---------- LLM (OpenAI) ----------
@dataclass
class LLMCloud:
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None

    def __post_init__(self):
        key = self.api_key or SETTINGS.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY ausente. Cole sua chave na sidebar ou configure o Secrets.")
        self.client = OpenAI(api_key=key)

    def answer(self, question: str, context: List[dict]) -> str:
        ctx = "\n\n".join(f"- {c['text']}" for c in context)
        prompt = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN.\n"
            "Responda apenas com base no contexto abaixo. Se faltar base, explique o que falta.\n\n"
            f"Pergunta: {question}\n\nContexto:\n{ctx}"
        )
        r = self.client.responses.create(model=self.model, input=prompt)
        return r.output_text

# ---------- Embeddings Local (CPU forçado – já estava ok) ----------
@dataclass
class EmbeddingsLocal:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name, device="cpu")
    def encode(self, texts: List[str]) -> np.ndarray:
        arr = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(arr, dtype="float32")
