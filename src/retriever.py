import chromadb
from .config import siliconflow_client, EMBED_MODEL

def get_collection(persist_dir="./index"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},
    )

def _embed_query(query: str) -> list[float]:
    resp = siliconflow_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    return resp.data[0].embedding

def retrieve(query: str, top_k: int = 4, source_filter: str = None):
    col = get_collection()
    query_vec = _embed_query(query)
    where = {"source": source_filter} if source_filter else None
    res = col.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
    )
    return [
        {"text": t, "source": m["source"], "distance": d}
        for t, m, d in zip(res["documents"][0],
                           res["metadatas"][0],
                           res["distances"][0])
    ]