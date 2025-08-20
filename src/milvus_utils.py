# src/milvus_utils.py
from typing import List
from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType
)
from .settings import SETTINGS

def connect() -> None:
    if connections.has_connection("default"):
        return
    secure = str(SETTINGS.milvus_uri).startswith("https")
    connections.connect(
        alias="default",
        uri=SETTINGS.milvus_uri,
        user=(SETTINGS.milvus_user or None),
        password=(SETTINGS.milvus_password or None),
        db_name=(SETTINGS.milvus_db or "default"),
        secure=secure,
        timeout=30,
    )

def get_or_create_collection(name: str, dim: int) -> Collection:
    connect()
    if utility.has_collection(name):
        col = Collection(name)
        col.load()
        return col

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="fonte", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="pagina", dtype=DataType.INT64),
        FieldSchema(name="tipo_licenca", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="tipo_empreendimento", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields, description="Docs NUPETR")
    col = Collection(name, schema)

    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}}
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()
    return col

def insert_records(col: Collection, embeddings, texts: List[str], fontes: List[str],
                   paginas: List[int], tipos_l: List[str], tipos_e: List[str]) -> None:
    # ordem segue o schema (id é auto)
    col.insert([embeddings, texts, fontes, paginas, tipos_l, tipos_e])
    col.flush()

def search(col: Collection, query_vec, top_k: int = 5, expr: str | None = None):
    params = {"metric_type": "IP", "params": {"nprobe": 32}}
    res = col.search(
        data=[query_vec],
        anns_field="embedding",
        param=params,
        limit=top_k,
        expr=expr,
        output_fields=["text", "fonte", "pagina", "tipo_licenca", "tipo_empreendimento"],
    )
    return res[0] if res else []

def drop_collection(name: str) -> None:
    connect()
    if utility.has_collection(name):
        utility.drop_collection(name)
