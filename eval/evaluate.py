import json
import sys
import os

# 把项目根目录加入路径，确保能引用到 src 里的 retriever
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retriever import retrieve

def evaluate(top_k=4):
    # 读取测试集
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    hit = 0
    print(f"🚀 开始评估，共 {len(cases)} 个测试问题，Top-K={top_k}\n")
    
    for c in cases:
        # 检索
        hits = retrieve(c["q"], top_k=top_k)
        sources = [h["source"] for h in hits]
        
        # 判断预期来源是否在其中
        if c["expected_source"] in sources:
            hit += 1
            print(f"✅ [命中] {c['q']}")
        else:
            print(f"❌ [未命中] {c['q']} → 实际检索到: {sources}")
            
    recall = hit / len(cases)
    print(f"\n📊 最终结果: Recall@{top_k} = {hit}/{len(cases)} = {recall:.1%}")
    return recall

if __name__ == "__main__":
    evaluate(top_k=4)