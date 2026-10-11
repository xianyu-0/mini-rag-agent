import os
import chromadb
import httpx
from .config import siliconflow_client, EMBED_MODEL


# ============================================================
# Chroma 集合
# ============================================================
def get_collection(persist_dir="./index"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},
    )


# ============================================================
# 向量检索
# ============================================================
def _embed_query(query: str) -> list:
    resp = siliconflow_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    return resp.data[0].embedding


def retrieve(query: str, top_k: int = 4, source_filter: str = None):
    """纯向量检索"""
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


# ============================================================
# Rerank 精排（调用硅基流动 API）
# ============================================================
def _rerank(query: str, candidates: list, top_n: int = 4):
    """
    对候选文档做 Rerank
    - candidates: [{"text":..., "source":...}, ...]
    - 返回: 精排后的 top_n 条（保留原字段，新增 rerank_score）
    """
    if not candidates:
        return []

    url = "https://api.siliconflow.cn/v1/rerank"
    headers = {
        "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": query,
        "documents": [c["text"] for c in candidates],
        "top_n": top_n,
        "return_documents": False,
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        results = resp.json()["results"]
        # 按 relevance_score 降序
        results.sort(key=lambda x: -x["relevance_score"])

        final = []
        for r in results:
            idx = r["index"]
            doc = candidates[idx].copy()
            doc["rerank_score"] = r["relevance_score"]
            final.append(doc)
        return final
    except Exception as e:
        print(f"⚠️ Rerank 失败，降级为纯向量结果: {e}")
        return candidates[:top_n]


# ============================================================
# 对外接口：向量召回 Top-20 → Rerank → Top-K
# ============================================================
def advanced_retrieve(query: str, top_k: int = 4):
    """
    高级检索：向量召回 20 条 → Rerank 精排 → 取 Top-K
    """
    # 1. 向量检索，召回 20 条（广撒网）
    candidates = retrieve(query, top_k=20)
    if not candidates:
        return []
    # 2. Rerank 精排，取 Top-K
    return _rerank(query, candidates, top_n=top_k)