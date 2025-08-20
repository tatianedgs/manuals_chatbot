from typing import List, Dict, Any
from pymilvus import connections, db, FieldSchema, CollectionSchema, DataType, Collection, utility
from .settings import SETTINGS

# Campos do schema
# - id (auto)
# - embedding (vector)
# - text (str)
# - fonte (str)
# - pagina (int)
# - tipo_licenca (str)
# - tipo_empreendimento (str)

def connect() -> None:
    if not connections.has_connection("default"):
        secure = str(SETTINGS.milvus_uri).startswith("https")
        connections.connect(
            alias="default",
            uri=SETTINGS.milvus_uri,
            user=SETTINGS.milvus_user or None,
            password=SETTINGS.milvus_password or None,
            secure=secure,
            timeout=30,
        )

    # garante DB (Milvus 2.4+)
    try:
        db.create_database(SETTINGS.milvus_db)
    except Exception:
        pass
    db.using_database(SETTINGS.milvus_db)


def get_or_create_collection(name: str, dim: int) -> Collection:
    connect()
    if utility.has_collection(name):
        col = Collection(name)
        col.load()
        return col

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="fonte", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="pagina", dtype=DataType.INT64),
        FieldSchema(name="tipo_licenca", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="tipo_empreendimento", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields=fields, description="Docs NUPETR")
    col = Collection(name=name, schema=schema)

    # índice e parâmetros de busca (cosine via produto interno com vetores normalizados)
    col.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
    )
    col.load()
    return col


def drop_collection(name: str) -> None:
    connect()
    if utility.has_collection(name):
        utility.drop_collection(name)


def insert_records(col: Collection, rows: List[Dict[str, Any]]):
    # Milvus espera listas por coluna (na ordem dos campos sem o id auto)
    embeddings = [r["embedding"] for r in rows]
    texts = [r["text"] for r in rows]
    fontes = [r.get("fonte", "") for r in rows]
    paginas = [int(r.get("pagina", 0)) for r in rows]
    tipos_l = [r.get("tipo_licenca", "") for r in rows]
    tipos_e = [r.get("tipo_empreendimento", "") for r in rows]

    col.insert([embeddings, texts, fontes, paginas, tipos_l, tipos_e])
    col.flush()


def search(col: Collection, query_vec, top_k=5, expr: str | None = None):
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