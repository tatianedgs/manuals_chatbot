# src/llm_router.py

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
    api_key: str = "" # Adiciona a chave como um atributo
    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key) # Usa a chave passada
    def encode(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

# ... (o resto da classe EmbeddingsLocal fica igual)

# ===== LLM backends =====

@dataclass
class LLMCloud:
    model: str = "gpt-5"
    api_key: str = "" # Adiciona a chave como um atributo
    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key) # Usa a chave passada
    def answer(self, question: str, context_blocks: List[str]) -> str:
        SYSTEM_PROMPT = (
            "Você é um assistente para analistas do NUPETR/IDEMA-RN.\n"
            "Responda baseado no contexto (manuais internos). Indique os trechos utilizados.\n"
            "Se não houver base no material, diga claramente."
        )

        context_str = "\n\n".join([f"[Trecho {i+1}]\n{c}" for i, c in enumerate(context_blocks)])
        
        user_input = f"Pergunta:\n{question}\n\nContexto:\n{context_str}"
        
        try:
            # Essa linha já estava na versão anterior, mas vou corrigi-la. 
            # O client.responses.create não existe. A função correta é client.chat.completions.create
            # Como não tenho acesso a sua biblioteca, estou supondo a função correta
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
            )
            # A forma de acessar a resposta também pode mudar.
            return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM Cloud indisponível] {e}"

# ... (o resto do arquivo llm_router.py fica igual)
