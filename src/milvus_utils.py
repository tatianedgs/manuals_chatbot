from __future__ import annotations

from typing import Dict, Any, List, Sequence
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from .settings import SETTINGS


def _sanitize_uri(uri: str) -> str:
    """Remove :19530 se alguém colar o endpoint com porta (Zilliz Serverless não usa)."""
    if uri and uri.startswith("http") and uri.endswith(":19530"):
        return uri[:-6]
    return uri


def connect() -> None:
    """Abre (ou reutiliza) a conexão default com Zilliz Serverless usando TOKEN."""
    if connections.has_connection("default"):
        return

    uri = _sanitize_uri(SETTINGS.milvus_uri)
    token = SETTINGS.milvus_token

    if not uri:
        raise ValueError("MILVUS_URI ausente. Configure nos Secrets.")
    if not token:
        raise ValueError(
            "MILVUS_TOKEN ausente. Em Zilliz, copie o **API Key** completo (API Keys → … → View) e cole em MILVUS_TOKEN."
        )

    kwargs: Dict[str, Any] = {
        "alias": "default",
        "uri": uri,          # https://....cloud.zilliz.com
        "secure": True,      # TLS
        "token": token,      # API Key (Serverless)
        "timeout": 30,
    }
    connections.connect(**kwargs)


# ===== Schema com PK auto =====
def _schema(dim: int) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
            FieldSchema(name="fonte", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="pagina", dtype=DataType.INT64),
            FieldSchema(name="tipo_licenca", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="tipo_empreendimento", dtype=DataType.VARCHAR, max_length=64),
        ],
        description="Trechos de PDFs institucionais (NUPETR/IDEMA-RN)",
        enable_dynamic_field=False,
    )


def get_or_create_collection(name: str, dim: int) -> Collection:
    connect()
    if utility.has_collection(name):
        col = Collection(name)
    else:
        col = Collection(name=name, schema=_schema(dim))
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "AUTOINDEX", "metric_type": "IP"},
        )
    col.load()
    return col


def drop_collection(name: str) -> None:
    connect()
    if utility.has_collection(name):
        utility.drop_collection(name)


# ===== Inserção e Busca (mesma assinatura usada no seu RAG) =====
def _normalize_records(regs: Sequence[Dict[str, Any]]):
    embs, texts, fontes, paginas, tlic, temp = [], [], [], [], [], []
    for r in regs:
        embs.append(list(map(float, r["embedding"])))
        texts.append(str(r["text"]))
        fontes.append(str(r.get("fonte", "")))
        paginas.append(int(r.get("pagina", 0)))
        tlic.append(str(r.get("tipo_licenca", "")))
        temp.append(str(r.get("tipo_empreendimento", "")))
    return embs, texts, fontes, paginas, tlic, temp


def insert_records(col: Collection, *args) -> None:
    # Aceita lista de dicts ou 6 listas paralelas (compatível com seu rag.py)
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and args[0] and isinstance(args[0][0], dict):
        embs, texts, fontes, paginas, tlic, temp = _normalize_records(args[0])  # type: ignore
    elif len(args) == 6:
        embs, texts, fontes, paginas, tlic, temp = args  # type: ignore
    else:
        raise TypeError("insert_records: use lista de dicts OU 6 listas paralelas")
    col.insert([embs, texts, fontes, paginas, tlic, temp])
    col.flush()


def search(col: Collection, qvec, top_k: int = 5, expr: str | None = None):
    params = {"metric_type": "IP", "params": {"nprobe": 32}}
    res = col.search(
        data=[qvec],
        anns_field="embedding",
        param=params,
        limit=top_k,
        expr=expr,
        output_fields=["text", "fonte", "pagina", "tipo_licenca", "tipo_empreendimento"],
    )
    return res[0] if res else []
