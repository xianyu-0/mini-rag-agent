import json
from .config import siliconflow_client, CHAT_MODEL
from .tools import TOOLS, TOOL_MAP
from .memory import Memory

# 系统提示词：决定 Agent 的行为逻辑
SYSTEM_PROMPT = """你是一个个人知识库助手，工作流程：

1. 先判断问题类型：
   - 涉及个人文档/笔记/资料 → 必须调用 search_knowledge_base
   - 需要时间 → 调用 get_current_time
   - 需要精确计算 → 调用 calculator
   - 纯常识/闲聊 → 直接回答，不要调用工具
2. 检索结果不足时，可以换个关键词再检索一次。
3. 回答时用 [来源: 文件名] 标注引用出处。
4. 如果知识库中确实没有，明确说"知识库中未找到"，不要编造。
"""

def run_agent(user_input: str, memory: Memory, max_steps: int = 6) -> str:
    memory.add("user", user_input)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.get()

    for step in range(max_steps):
        # 让大模型思考（用硅基流动的 Qwen 模型）
        resp = siliconflow_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # 没有工具调用 → 得到最终答案，结束循环
        if not msg.tool_calls:
            memory.add("assistant", msg.content)
            return msg.content

        # 有工具调用 → 执行工具，把结果喂回给大模型
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"  🔧 调用工具: {fn_name}({args})")
            try:
                result = TOOL_MAP[fn_name](**args)
            except Exception as e:
                result = f"工具执行失败: {e}"
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)[:4000], # 截断防止上下文爆炸
            })

    return "⚠️ 达到最大步数限制，请换个问法。"