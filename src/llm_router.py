from typing import List, Optional
from dataclasses import dataclass
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from .settings import SETTINGS

# ===================== Embeddings =====================

@dataclass
class EmbeddingsCloud:
    model: str = "text-embedding-3-large"
    api_key: Optional[str] = None  # permite usar a chave digitada pelo usuário

    def __post_init__(self):
        key = self.api_key or SETTINGS.openai_api_key
        self.client = OpenAI(api_key=key)

    def encode(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

@dataclass
class EmbeddingsLocal:
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self):
        # força CPU para evitar erro "meta tensor" no Streamlit Cloud
        self.model = SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

# ===================== LLMs =====================

@dataclass
class LLMCloud:
    model: str = "gpt-5"   # alias lógico; ajuste para o modelo correto do seu plano
    api_key: Optional[str] = None

    def __post_init__(self):
        key = self.api_key or SETTINGS.openai_api_key
        self.client = OpenAI(api_key=key)

    def answer(self, question: str, context_blocks: List[str]) -> str:
        system = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN. "
            "Responda com base APENAS nos trechos de contexto (manuais internos). "
            "Se não houver base suficiente, diga claramente."
        )
        ctx = "\n\n".join([f"[Trecho {i+1}]\n{c}" for i, c in enumerate(context_blocks)])
        user = f"Pergunta:\n{question}\n\nContexto:\n{ctx}"

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.output_text
        except Exception as e:
            return f"[LLM Cloud indisponível] {e}"

@dataclass
class LLMLocal:
    model: str = SETTINGS.ollama_model
    host: str = SETTINGS.ollama_host

    def answer(self, question: str, context_blocks: List[str]) -> str:
        url = f"{self.host}/api/generate"
        ctx = "\n\n".join([f"[Trecho {i+1}]\n{c}" for i, c in enumerate(context_blocks)])
        prompt = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN.\n"
            "Responda com base APENAS no contexto. Se faltar base, diga.\n\n"
            f"Pergunta: {question}\n\nContexto:\n{ctx}"
        )
        try:
            r = requests.post(url, json={"model": self.model, "prompt": prompt, "stream": False}, timeout=120)
            r.raise_for_status()
            j = r.json()
            return j.get("response", "(sem resposta do modelo local)")
        except Exception as e:
            return f"[LLM local indisponível] {e}"
