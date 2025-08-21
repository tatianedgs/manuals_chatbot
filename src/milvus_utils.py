from __future__ import annotations

from typing import List, Dict, Any, Sequence
from pymilvus import (
    connections,
    db,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from .settings import SETTINGS


# =========================================================
# Conexão
# =========================================================
def connect() -> None:
    """Abre (ou reutiliza) a conexão default com o Milvus/Zilliz.

    - Serverless (TOKEN): usa `token`, `secure=True` e **NÃO** passa db_name.
    - Dedicated (user/senha): passa `user/password` e `db_name` e pode criar DB.
    """
    if connections.has_connection("default"):
        return

    kwargs: Dict[str, Any] = {
        "alias": "default",
        "uri": SETTINGS.milvus_uri,
        "timeout": 30,
    }

    # HTTPS/TLS no Zilliz
    if str(SETTINGS.milvus_uri).startswith("https"):
        kwargs["secure"] = True

    if SETTINGS.milvus_token:
        # ---- SERVERLESS (API Key / Token) ----
        kwargs["token"] = SETTINGS.milvus_token
        # NÃO passar db_name e NÃO criar database em Serverless
    else:
        # ---- DEDICATED (usuário/senha) ----
        if SETTINGS.milvus_user:
            kwargs["user"] = SETTINGS.milvus_user
        if SETTINGS.milvus_password:
            kwargs["password"] = SETTINGS.milvus_password
        if SETTINGS.milvus_db:
            kwargs["db_name"] = SETTINGS.milvus_db

    connections.connect(**kwargs)

    # Criar/selecionar DB só em Dedicated
    if not SETTINGS.milvus_token and SETTINGS.milvus_db:
        try:
            db.create_database(SETTINGS.milvus_db)
        except Exception:
            pass
        try:
            db.using_database(SETTINGS.milvus_db)
        except Exception:
            pass


# =========================================================
# Esquema e coleção
# =========================================================
def _schema(dim: int) -> CollectionSchema:
    return CollectionSchema(
        fields=[
            # >>> Primary Key (gerado automaticamente pelo servidor)
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

            # Demais campos
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


# =========================================================
# Inserção & Busca
# =========================================================
def _normalize_records(registros: Sequence[Dict[str, Any]]):
    embeddings: List[List[float]] = []
    texts: List[str] = []
    fontes: List[str] = []
    paginas: List[int] = []
    tipos_l: List[str] = []
    tipos_e: List[str] = []
    for r in registros:
        embeddings.append(list(map(float, r["embedding"])))
        texts.append(str(r["text"]))
        fontes.append(str(r.get("fonte", "")))
        paginas.append(int(r.get("pagina", 0)))
        tipos_l.append(str(r.get("tipo_licenca", "")))
        tipos_e.append(str(r.get("tipo_empreendimento", "")))
    return embeddings, texts, fontes, paginas, tipos_l, tipos_e


def insert_records(col: Collection, *args) -> None:
    # Formato 1: lista de dicts
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and args[0] and isinstance(args[0][0], dict):
        embeddings, texts, fontes, paginas, tipos_l, tipos_e = _normalize_records(args[0])  # type: ignore
    # Formato 2: 6 listas paralelas
    elif len(args) == 6:
        embeddings, texts, fontes, paginas, tipos_l, tipos_e = args  # type: ignore
    else:
        raise TypeError("insert_records: use lista de dicts OU 6 listas (embeddings, texts, fontes, paginas, tipos_l, tipos_e)")
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
