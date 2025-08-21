# src/pdf_utils.py
from __future__ import annotations
import io
import re
from typing import Iterable, Iterator, Tuple
from pypdf import PdfReader

def extract_text_pages(file_bytes: bytes, fonte: str) -> Iterator[Tuple[str, int, str]]:
    """
    Extrai texto página a página de um PDF e retorna (texto_da_pagina, numero_pagina, fonte).
    A assinatura bate com o rag.py.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        # normaliza espaços
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        yield txt, i, fonte

def chunk_text(texto: str, max_chars: int = 1200, overlap: int = 200) -> Iterator[str]:
    """
    Quebra o texto em pedaços com sobreposição para caber em limites de embedding/VARCHAR.
    O rag.py ainda reforça um corte em 16k antes de inserir.
    """
    if not texto:
        return
    n = len(texto)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        yield texto[start:end]
        if end == n:
            break
        start = max(0, end - overlap)
