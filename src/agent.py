import json
from .config import siliconflow_client, CHAT_MODEL
from .tools import TOOLS, TOOL_MAP


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


def run_agent_stream(user_input: str, memory, max_steps: int = 6):
    """
    流式输出版本的 Agent 主循环（ReAct）
    - 每生成一个 token 就 yield 出来
    - 支持 Function Calling（工具调用）
    """
    memory.add("user", user_input)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.get()

    for step in range(max_steps):
        # 1. 开启流式调用
        stream = siliconflow_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            stream=True,
        )

        assistant_text = ""
        tool_calls = []  # 累积工具调用（分片到达）

        # 2. 逐 chunk 处理
        for chunk in stream:
            delta = chunk.choices[0].delta

            # 文本内容：实时 yield
            if delta.content:
                assistant_text += delta.content
                yield delta.content

            # 工具调用：分片累积
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    while len(tool_calls) <= idx:
                        tool_calls.append({"id": "", "name": "", "arguments": ""})
                    if tc_delta.id:
                        tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls[idx]["arguments"] += tc_delta.function.arguments

        # 3. 组装 assistant 消息
        assistant_msg = {"role": "assistant", "content": assistant_text or None}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        # 4. 没有工具调用 → 结束
        if not tool_calls:
            memory.add("assistant", assistant_text)
            return

        # 5. 执行工具，把结果喂回去
        for tc in tool_calls:
            fn_name = tc["name"]
            args = json.loads(tc["arguments"] or "{}")
            yield f"\n\n> 🔧 正在调用工具：`{fn_name}`\n\n"
            try:
                result = TOOL_MAP[fn_name](**args)
            except Exception as e:
                result = f"工具执行失败: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result)[:4000],
            })

    yield "\n\n⚠️ 达到最大步数限制，请换个问法。"