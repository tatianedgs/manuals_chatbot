from typing import Iterable, List, Dict
import numpy as np
from .milvus_utils import get_or_create_collection, insert_records, search
from .pdf_utils import extract_text_pages, chunk_text

# ===== Ingestão =====

def build_embeddings(encoder, texts: List[str]) -> np.ndarray:
    return encoder.encode(texts)


def ingest_pdfs(encoder, files: Iterable[tuple[str, bytes]], tipo_licenca: str, tipo_empreendimento: str, collection_name: str) -> int:
    col = get_or_create_collection(collection_name, dim=encoder.encode(["test"]).shape[1])
    to_insert: List[Dict] = []

    for fname, fbytes in files:
        pages = extract_text_pages(fbytes, fonte=fname)
        for page_text, pagina, fonte in pages:
            for ch in chunk_text(page_text):
                to_insert.append({
                    "text": ch,
                    "fonte": fonte,
                    "pagina": pagina,
                    "tipo_licenca": tipo_licenca.upper().strip(),
                    "tipo_empreendimento": tipo_empreendimento.upper().strip(),
                })

    B = 128
    total = 0
    for i in range(0, len(to_insert), B):
        batch = to_insert[i:i+B]
        vecs = build_embeddings(encoder, [r["text"] for r in batch])
        for j, v in enumerate(vecs):
            batch[j]["embedding"] = v.tolist()
        insert_records(col, batch)
        total += len(batch)

    return total

# ===== Busca =====

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