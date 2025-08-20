# src/rag.py
from typing import Iterable, List, Dict
import numpy as np
from .milvus_utils import get_or_create_collection, insert_records, search
from .pdf_utils import extract_text_pages, chunk_text

def build_embeddings(encoder, texts: List[str]) -> np.ndarray:
    return encoder.encode(texts)

def ingest_pdfs(
    encoder,
    files: Iterable[tuple[str, bytes]],
    tipo_licenca: str,
    tipo_empreendimento: str,
    collection_name: str,
) -> int:
    """Quebra PDFs em trechos, gera embeddings e grava no Milvus."""
    col = get_or_create_collection(collection_name, dim=encoder.encode(["test"]).shape[1])
    registros: List[Dict] = []
    n_chunks = 0

    for fname, fbytes in files:
        # 'fonte' (como aparece nas citações)
        fonte = f"{tipo_licenca}_{tipo_empreendimento}_{fname}"

        # AGORA passamos 'fonte' para a função de extração
        for texto_pagina, pagina, _fonte in extract_text_pages(fbytes, fonte=fonte):
            # divide em pedaços para caber no embedding/consulta
            for trecho in chunk_text(texto_pagina):
                emb = encoder.encode([trecho])[0].tolist()
                registros.append({
                    "embedding": emb,
                    "text": trecho,
                    "fonte": fonte,
                    "pagina": int(pagina),
                    "tipo_licenca": tipo_licenca,
                    "tipo_empreendimento": tipo_empreendimento,
                })
                n_chunks += 1

    if registros:
        insert_records(col, registros)

    return n_chunks

def retrieve_top_k(encoder, query: str, collection_name: str, top_k: int = 5, expr: str | None = None):
    col = get_or_create_collection(collection_name, dim=encoder.encode(["test"]).shape[1])
    q = encoder.encode([query])[0]
    hits = search(col, q.tolist(), top_k=top_k, expr=expr)
    out = []
    for h in hits:
        out.append({
            "score": float(h.distance),
            "text": h.entity.get("text"),
            "fonte": h.entity.get("fonte"),
            "pagina": int(h.entity.get("pagina")),
            "tipo_licenca": h.entity.get("tipo_licenca"),
            "tipo_empreendimento": h.entity.get("tipo_empreendimento"),
        })
    return out
