import io
from typing import List, Tuple
from pypdf import PdfReader


def extract_text_pages(file_bytes: bytes, fonte: str) -> List[Tuple[str, int, str]]:
    """Retorna lista de (texto, pagina, fonte)."""
    r = PdfReader(io.BytesIO(file_bytes))
    out = []
    for i, p in enumerate(r.pages, start=1):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            out.append((txt, i, fonte))
    return out


def chunk_text(text: str, max_chars=1800, overlap=200) -> list[str]:
    text = text.replace("�", " ")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]