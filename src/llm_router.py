# src/llm_router.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from .settings import SETTINGS

# ========= Embeddings =========

@dataclass
class EmbeddingsCloud:
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None

    def __post_init__(self):
        key = self.api_key or SETTINGS.openai_api_key
        self.client = OpenAI(api_key=key)

    def encode(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype=np.float32)

@dataclass
class EmbeddingsLocal:
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self):
        # força CPU para evitar o "NotImplementedError: meta..." no Cloud
        self.model = SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts: List[str]) -> np.ndarray:
        arr = self.model.encode(texts, normalize_embeddings=True)
        return np.array(arr, dtype=np.float32)

# ========= LLMs =========

@dataclass
class LLMCloud:
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None

    def __post_init__(self):
        key = self.api_key or SETTINGS.openai_api_key
        self.client = OpenAI(api_key=key)

    def answer(self, question: str, context: str) -> str:
        msgs = [
            {"role": "system",
             "content": ("Você é um assistente para analistas do NUPETR/IDEMA-RN. "
                         "Responda APENAS com base no contexto fornecido. "
                         "Se não houver base suficiente, diga claramente o que falta.")},
            {"role": "user", "content": f"Pergunta: {question}\n\nContexto:\n{context}"}
        ]
        out = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.2)
        return out.choices[0].message.content.strip()

@dataclass
class LLMLocal:
    model: str = SETTINGS.ollama_model
    host: str = SETTINGS.ollama_host

    def answer(self, question: str, context: str) -> str:
        url = f"{self.host}/api/generate"
        prompt = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN.\n"
            "Responda APENAS com base no contexto. Se faltar base, diga.\n\n"
            f"Pergunta: {question}\n\nContexto:\n{context}"
        )
        try:
            r = requests.post(url, json={"model": self.model, "prompt": prompt, "stream": False}, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            return f"[LLM local indisponível] {e}"
