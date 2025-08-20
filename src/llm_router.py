from typing import List
from dataclasses import dataclass
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from .settings import SETTINGS

# ========== Embeddings ==========

@dataclass
class EmbeddingsCloud:
    model: str = "text-embedding-3-large"
    def __post_init__(self):
        self.client = OpenAI(api_key=SETTINGS.openai_api_key)
    def encode(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.asarray(vecs, dtype="float32")

@dataclass
class EmbeddingsLocal:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    def __post_init__(self):
        # força CPU e evita meta tensors do PyTorch
        self.model = SentenceTransformer(self.model_name, device="cpu")
    def encode(self, texts: List[str]) -> np.ndarray:
        arr = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(arr, dtype="float32")

# ========== LLMs ==========

@dataclass
class LLMCloud:
    model: str = "gpt-4o-mini"  # troque se quiser outro id disponível na sua conta
    def __post_init__(self):
        self.client = OpenAI(api_key=SETTINGS.openai_api_key)
    def answer(self, question: str, context: List[dict]) -> str:
        ctx = "\n\n".join(f"- {c['text']}" for c in context)
        prompt = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN.\n"
            "Responda apenas com base no contexto abaixo. Se faltar base, explique o que falta.\n\n"
            f"Pergunta: {question}\n\nContexto:\n{ctx}"
        )
        r = self.client.responses.create(model=self.model, input=prompt)
        return r.output_text

@dataclass
class LLMLocal:
    model: str | None = None  # usa SETTINGS.ollama_model se None
    def answer(self, question: str, context: List[dict]) -> str:
        model = self.model or SETTINGS.ollama_model
        url = f"{SETTINGS.ollama_host}/api/generate"
        ctx = "\n\n".join(f"- {c['text']}" for c in context)
        prompt = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN.\n"
            "Responda apenas com base no contexto abaixo. Se faltar base, explique o que falta.\n\n"
            f"Pergunta: {question}\n\nContexto:\n{ctx}"
        )
        try:
            j = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=120).json()
            return j.get("response", "(sem resposta do modelo local)")
        except Exception as e:
            return f"[LLM local indisponível] {e}"
