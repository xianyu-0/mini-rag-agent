import json
import sys
import os

# 把项目根目录加入路径，确保能引用到 src 里的 retriever
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retriever import advanced_retrieve


def evaluate(top_k=4):
    # 读取测试集
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    hit = 0
    print(f"🚀 开始评估，共 {len(cases)} 个测试问题，Top-K={top_k}\n")

    for c in cases:
        # 用高级检索（向量召回 + Rerank 精排）
        hits = advanced_retrieve(c["q"], top_k=top_k)
        sources = [h["source"] for h in hits]

        # ============================================================
        # 命中判断：支持单个来源（字符串）和多个来源（数组）
        # ============================================================
        expected = c["expected_source"]
        if isinstance(expected, list):
            # 多来源：命中任意一个就算对
            matched = any(e in sources for e in expected)
        else:
            # 单来源：必须命中
            matched = expected in sources

        if matched:
            hit += 1
            print(f"✅ [命中] {c['q']}")
        else:
            print(f"❌ [未命中] {c['q']} → 实际检索到: {sources}")

    recall = hit / len(cases)
    print(f"\n📊 最终结果: Recall@{top_k} = {hit}/{len(cases)} = {recall:.1%}")
    return recall


if __name__ == "__main__":
    evaluate(top_k=4)