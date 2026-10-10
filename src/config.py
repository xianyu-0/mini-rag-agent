import os
from openai import OpenAI

# ============================================================
# 硅基流动：既负责对话生成，也负责 Embedding 向量化
# （因为 DeepSeek 没充值，且硅基流动有免费额度）
# ============================================================
siliconflow_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

# 对话模型：用硅基流动上的免费 Qwen 模型（效果很好，且免费）
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 向量模型：免费 bge-m3
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

INDEX_DIR = "./index"
DATA_DIR  = "./data"