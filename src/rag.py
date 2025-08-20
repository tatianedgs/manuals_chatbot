from typing import Iterable, List, Dict, Tuple
import numpy as np
from .milvus_utils import get_or_create_collection, insert_records, search
from .pdf_utils import extract_text_pages, chunk_text

def build_embeddings(encoder, texts: List[str]) -> np.ndarray:
    return encoder.encode(texts)

def ingest_pdfs(
    encoder,
    files: Iterable[Tuple[str, bytes]],
    tipo_licenca: str,
    tipo_empreendimento: str,
    collection_name: str,
) -> int:
    col = get_or_create_collection(collection_name, dim=encoder.encode(["test"]).shape[1])
    to_insert: List[Dict] = []
    for fname, fbytes in files:
        pages = extract_text_pages(fbytes)
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text, chunk_size=800, overlap=120):
                to_insert.append({
                    "text": chunk,
                    "fonte": fname,
                    "pagina": page_num,
                    "tipo_licenca": tipo_licenca,
                    "tipo_empreendimento": tipo_empreendimento,
                })
    if not to_insert:
        return 0
    embs = build_embeddings(encoder, [r["text"] for r in to_insert])
    for i, r in enumerate(to_insert):
        r["embedding"] = embs[i].tolist()
    insert_records(col, to_insert)
    return len(to_insert)

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
