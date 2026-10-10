import os, glob, chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import siliconflow_client, EMBED_MODEL

def load_docs(path="./data"):
    docs = []
    for f in glob.glob(f"{path}/**/*.md", recursive=True) + \
             glob.glob(f"{path}/**/*.txt", recursive=True):
        with open(f, encoding="utf-8") as fp:
            docs.append({"text": fp.read(), "source": os.path.basename(f)})
    print(f"📄 加载了 {len(docs)} 篇文档")
    return docs

def chunk_docs(docs, chunk_size=500, overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " "]
    )
    chunks = []
    for d in docs:
        for i, c in enumerate(splitter.split_text(d["text"])):
            chunks.append({
                "id": f"{d['source']}::{i}",
                "text": c,
                "source": d["source"],
            })
    print(f"✂️ 切分成了 {len(chunks)} 个片段")
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = siliconflow_client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        all_embeddings.extend([d.embedding for d in resp.data])
        print(f"  ⏳ 已向量化 {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings

def build_index(chunks, persist_dir="./index"):
    client = chromadb.PersistentClient(path=persist_dir)
    print("🧮 开始调用硅基流动 bge-m3 向量化...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    col = client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"},
    )
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"✅ 入库 {len(chunks)} 个分块！")

if __name__ == "__main__":
    docs = load_docs()
    chunks = chunk_docs(docs)
    build_index(chunks)