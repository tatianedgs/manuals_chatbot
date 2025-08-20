from typing import List
from dataclasses import dataclass
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from .settings import SETTINGS

# ===== Embeddings backends =====

@dataclass
class EmbeddingsCloud:
    model: str = "text-embedding-3-large"
    api_key: str = ""
    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key)
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
        self.model = SentenceTransformer(self.model_name)
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype("float32")

# ===== LLM backends =====

@dataclass
class LLMCloud:
    model: str = "gpt-5"
    api_key: str = ""

    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key)

    def answer(self, question: str, context_blocks: List[str]) -> str:
        SYSTEM_PROMPT = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN.\n"
            "Responda baseado no contexto (manuais internos). Indique os trechos utilizados.\n"
            "Se não houver base no material, diga claramente."
        )

        context_str = "\n\n".join([f"[Trecho {i+1}]\n{c}" for i, c in enumerate(context_blocks)])
        
        user_input = f"Pergunta:\n{question}\n\nContexto:\n{context_str}"
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM Cloud indisponível] {e}"

@dataclass
class LLMLocal:
    model: str = SETTINGS.ollama_model
    host: str = SETTINGS.ollama_host

    def answer(self, question: str, context_blocks: List[str]) -> str:
        url = f"{self.host}/api/generate"
        
        context_str = "\n\n".join([f"[Trecho {i+1}]\n{c}" for i, c in enumerate(context_blocks)])
        
        prompt = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN.\n"
            "Responda com base APENAS no contexto. Se faltar base, diga.\n\n"
            f"Pergunta: {question}\n\nContexto:\n{context_str}"
        )

        data = {"model": self.model, "prompt": prompt, "stream": False}
        
        try:
            r = requests.post(url, json=data, timeout=120)
            r.raise_for_status()
            j = r.json()
            return j.get("response", "(sem resposta do modelo local)")
        except Exception as e:
            return f"[LLM local indisponível] {e}"
