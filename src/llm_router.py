# src/llm_router.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence

# Embeddings cloud/local
from openai import OpenAI

try:
    # sentence-transformers para embeddings locais (CPU)
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Para o extrativo (sem LLM)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# -------------------------
# Util
# -------------------------
def _ensure_2d_list(vecs) -> List[List[float]]:
    if hasattr(vecs, "tolist"):
        vecs = vecs.tolist()
    if isinstance(vecs, list) and vecs and isinstance(vecs[0], (int, float)):
        return [list(map(float, vecs))]
    return [[float(x) for x in row] for row in vecs]


def _split_sentences(text: str) -> List[str]:
    # split simples e robusto, evitando quebrar demais
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÂÃÀÉÊÍÓÔÕÚÜÇ])", text)
    # fallback se vier muito curto
    if len(sents) <= 1:
        sents = re.split(r"[.;:\n]+", text)
    return [s.strip() for s in sents if s.strip()]


# -------------------------
# Embeddings
# -------------------------
@dataclass
class EmbeddingsCloud:
    model: str = "text-embedding-3-large"  # 3072 dims

    def __post_init__(self):
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY ausente para EmbeddingsCloud.")
        self.client = OpenAI(api_key=key)

    def encode(self, texts: Sequence[str]):
        # OpenAI v1
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        vecs = [d.embedding for d in resp.data]
        return _ensure_2d_list(vecs)


@dataclass
class EmbeddingsLocal:
    model_name: str = "all-MiniLM-L6-v2"  # 384 dims (rápido)

    def __post_init__(self):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers não disponível para EmbeddingsLocal.")
        # força CPU
        self.model = SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts: Sequence[str]):
        vecs = self.model.encode(list(texts), show_progress_bar=False, batch_size=64, convert_to_numpy=True)
        return _ensure_2d_list(vecs)


# -------------------------
# LLM na Nuvem (opcional)
# -------------------------
@dataclass
class LLMCloud:
    model: str = "gpt-4o-mini"

    def __post_init__(self):
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY ausente para LLMCloud.")
        self.client = OpenAI(api_key=key)

    def answer(self, question: str, contexts: Sequence[str]) -> str:
        ctx = "\n\n---\n\n".join(contexts[:8]) if contexts else "N/A"
        prompt = (
            "Você é um analista técnico do IDEMA/RN.\n"
            "Responda de forma objetiva, citando apenas informações presentes no CONTEXTO abaixo.\n"
            "Se algo não estiver no contexto, diga que não há dados suficientes.\n\n"
            f"PERGUNTA:\n{question}\n\n"
            f"CONTEXTO:\n{ctx}\n"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


# -------------------------
# LiteLocal — resposta EXTRATIVA (sem LLM)
# -------------------------
class LiteLocal:
    """
    Módulo de resposta extrativa:
    - Não "inventa" texto. Só seleciona/fraciona frases que melhor respondem,
      a partir dos trechos recuperados (RAG).
    - Implementado com TF-IDF + similaridade cosseno.
    """

    def __init__(self, max_sentences: int = 6):
        self.max_sentences = max_sentences

    def answer(self, question: str, contexts: Sequence[str]) -> str:
        if not contexts:
            return "Não encontrei trechos suficientes no acervo para responder."

        # Junta todas as frases candidatas
        candidates: List[str] = []
        for c in contexts:
            candidates.extend(_split_sentences(c)[:20])  # corta por contexto para evitar explosão

        # TF-IDF: pergunta vs frases
        docs = [question] + candidates
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000).fit_transform(docs)
        q_vec = tfidf[0:1]
        sents_vec = tfidf[1:]
        sims = linear_kernel(q_vec, sents_vec).flatten()

        # Top-k frases distintas
        idxs = sims.argsort()[::-1]
        picked, seen = [], set()
        for i in idxs:
            sent = candidates[i].strip()
            key = sent.lower()
            if key not in seen:
                picked.append(sent)
                seen.add(key)
            if len(picked) >= self.max_sentences:
                break

        if not picked:
            return "Encontrei trechos, mas nenhum responde claramente à pergunta."

        # Monta resposta extrativa
        bullets = "\n".join([f"• {s}" for s in picked])
        return f"**Resposta extrativa (sem LLM):**\n\n{bullets}"
