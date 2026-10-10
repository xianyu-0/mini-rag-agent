from src.retriever import retrieve

if __name__ == "__main__":
    q = "RAG 的核心流程是什么"  # 这里填你刚才放进 test.txt 里的话题
    for h in retrieve(q, top_k=2):
        print(f"[{h['source']}] dist={h['distance']:.3f}")
        print(h['text'][:200], "...\n")