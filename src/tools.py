from datetime import datetime
from .retriever import retrieve

def search_knowledge_base(query: str) -> str:
    # 在个人知识库中检索相关文档片段。
    hits = retrieve(query, top_k=4)
    if not hits:
        return "知识库中未找到相关内容。"
    return "\n\n".join(f"[来源: {h['source']}]\n{h['text']}" for h in hits)

def get_current_time() -> str:
    # 获取当前日期时间。
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculator(expression: str) -> str:
    # 计算数学表达式，如 '2**10 + 5'。
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败: {e}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "当用户问题涉及个人文档、笔记、技术资料时调用。纯常识问题（如'你好'、'1+1'）不要调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索用的关键词或问题"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "需要当前日期或时间时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "需要精确数学计算时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_MAP = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "calculator": calculator,
}