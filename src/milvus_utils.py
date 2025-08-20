from typing import List, Dict, Any
from pymilvus import connections, db, FieldSchema, CollectionSchema, DataType, Collection, utility
from .settings import SETTINGS

def connect() -> None:
    """Conecta ao Milvus/Zilliz usando token (ou user/senha)."""
    if connections.has_connection("default"):
        return

    kwargs = dict(
        alias="default",
        uri=SETTINGS.milvus_uri,
        db_name=SETTINGS.milvus_db,
        timeout=30,
    )

    # HTTPS no Zilliz Cloud
    if str(SETTINGS.milvus_uri).startswith("https"):
        kwargs["secure"] = True

    # Preferir TOKEN (Zilliz). Se não houver, tentar user/senha.
    if SETTINGS.milvus_token:
        kwargs["token"] = SETTINGS.milvus_token
    else:
        if SETTINGS.milvus_user:
            kwargs["user"] = SETTINGS.milvus_user
        if SETTINGS.milvus_password:
            kwargs["password"] = SETTINGS.milvus_password

    connections.connect(**kwargs)

    # Cria o DB se não existir (ignora erro se já existe)
    try:
        db.create_database(SETTINGS.milvus_db)
    except Exception:
        pass

def get_or_create_collection(name: str, dim: int) -> Collection:
    connect()
    if not utility.has_collection(name):
        fields = [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="fonte", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="pagina", dtype=DataType.INT64),
            FieldSchema(name="tipo_licenca", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="tipo_empreendimento", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields, description="docs NUPETR")
        col = Collection(name=name, schema=schema)
        col.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}})
    else:
        col = Collection(name)
    return col

def insert_records(col: Collection, records: List[Dict[str, Any]]):
    embeddings = [r["embedding"] for r in records]
    texts = [r["text"] for r in records]
    fontes = [r["fonte"] for r in records]
    paginas = [int(r["pagina"]) for r in records]
    tipos_l = [r.get("tipo_licenca", "") for r in records]
    tipos_e = [r.get("tipo_empreendimento", "") for r in records]
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

def drop_collection(name: str):
    connect()
    if utility.has_collection(name):
        utility.drop_collection(name)
