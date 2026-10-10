from src.agent import run_agent
from src.memory import Memory

m = Memory()

print("--- 测试 1：涉及知识库（应调用检索） ---")
print(run_agent("RAG 的核心流程是什么？", m))

print("\n--- 测试 2：需要工具（应调用时间） ---")
print(run_agent("现在几点了？", m))

print("\n--- 测试 3：纯闲聊（不该调用工具） ---")
print(run_agent("你好呀，介绍一下你自己", m))