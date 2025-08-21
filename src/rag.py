# src/rag.py
from __future__ import annotations

from typing import Iterable, List, Dict, Tuple
import numpy as np

from .milvus_utils import get_or_create_collection, insert_records, search
from .pdf_utils import extract_text_pages, chunk_text

# Tamanho máximo para caber no VARCHAR(16384) com folga
MAX_CHARS = 16000
# Tamanho de lote para gerar embeddings (evita milhares de chamadas)
BATCH_SIZE = 64


def _to_2d_array(x) -> np.ndarray:
    """Garante array 2D float32."""
    arr = np.array(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _embed_batch(encoder, texts: List[str]) -> np.ndarray:
    """Encapsula o encoder.encode, garantindo np.ndarray float32."""
    vecs = encoder.encode(texts)
    return _to_2d_array(vecs)


def _probe_dim(encoder) -> int:
    """Descobre a dimensão do embedding no runtime."""
    v = _embed_batch(encoder, ["__probe__"])
    return int(v.shape[1])


def ingest_pdfs(
    encoder,
    files: Iterable[Tuple[str, bytes]],
    tipo_licenca: str,
    tipo_empreendimento: str,
    collection_name: str,
) -> int:
    """
    Lê PDFs, quebra em páginas/trechos, gera embeddings e grava no Milvus.
    Retorna o número de trechos inseridos.
    """
    dim = _probe_dim(encoder)
    col = get_or_create_collection(collection_name, dim=dim)

    # Colete todos os trechos primeiro (para batch de embeddings)
    pending: List[Tuple[str, str, int, str, str]] = []
    #            (texto, fonte, pagina, tipo_licenca, tipo_empreendimento)

    for fname, fbytes in files:
        fonte = f"{tipo_licenca}_{tipo_empreendimento}_{fname}"
        # extract_text_pages deve render (texto_pagina, pagina, fonte)
        for texto_pagina, pagina, _fonte in extract_text_pages(fbytes, fonte=fonte):
            # quebra a página em pedaços menores
            for trecho in chunk_text(texto_pagina):
                texto = (trecho or "").strip()
                if not texto:
                    continue
                # corte de segurança para caber no VARCHAR
                texto = texto[:MAX_CHARS]
                pending.append((texto, fonte, int(pagina), tipo_licenca, tipo_empreendimento))

    if not pending:
        return 0

    # Embeddings em lote e inserção
    registros: List[Dict] = []
    total = 0
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i : i + BATCH_SIZE]
        texts = [t[0] for t in batch]
        vecs = _embed_batch(encoder, texts)
        for j, (texto, fonte, pagina, tlic, temp) in enumerate(batch):
            emb_j = vecs[j].tolist()
            registros.append(
                {
                    "embedding": emb_j,
                    "text": texto,
                    "fonte": fonte,
                    "pagina": int(pagina),
                    "tipo_licenca": tlic,
                    "tipo_empreendimento": temp,
                }
            )
            total += 1

    if registros:
        insert_records(col, registros)

    return total


def retrieve_top_k(
    encoder,
    query: str,
    collection_name: str,
    top_k: int = 5,
    expr: str | None = None,
):
    """Faz a busca vetorial (com filtro opcional) e retorna hits em dicts simples."""
    dim = _probe_dim(encoder)
    col = get_or_create_collection(collection_name, dim=dim)

    qvec = _embed_batch(encoder, [query])[0].tolist()
    result = search(col, qvec, top_k=top_k, expr=expr)

    out = []
    for h in result:
        # h.distance e h.entity[...] (PyMilvus Hit)
        score = float(getattr(h, "distance", 0.0))
        entity = getattr(h, "entity", None)
        if entity is None:  # fallback defensivo
            continue
        out.append(
            {
                "score": score,
                "text": entity.get("text"),
                "fonte": entity.get("fonte"),
                "pagina": int(entity.get("pagina")),
                "tipo_licenca": entity.get("tipo_licenca"),
                "tipo_empreendimento": entity.get("tipo_empreendimento"),
            }
        )
    return out
