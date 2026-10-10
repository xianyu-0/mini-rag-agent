# 📚 个人知识库 Agent（mini-rag-agent）

一个基于 **Agentic RAG（自主检索增强生成）** 的个人知识库问答助手。不仅能检索本地文档，还能通过大模型自主判断何时该检索、何时该调用工具、何时该直接回答。

## 🎯 核心亮点

- **Agentic RAG**：不强制每次检索，大模型自主判断意图，节省 Token 且更智能。
- **Function Calling**：支持检索知识库、获取当前时间、执行数学计算三种工具。
- **多轮记忆**：支持追问，能结合上一轮对话继续检索。
- **量化评估**：用测试集验证检索质量，Recall@4 达到 **100%**。

## 🏗️ 系统架构

```text
用户提问 → Agent 主循环 (ReAct) → 大模型自主决策
                                  ├─ 闲聊 → 直接回答
                                  ├─ 知识库问题 → search_knowledge_base
                                  ├─ 时间问题 → get_current_time
                                  └─ 计算问题 → calculator
                                        ↓
                                  Chroma 向量库
                                        ↓
                     离线阶段：文档 → 切分 → bge-m3 向量化 → 入库
🛠️ 技术栈
组件	选型	说明
对话大模型	Qwen2.5-7B	通过硅基流动 API 调用
Embedding	BAAI/bge-m3	免费、中文强，1024 维
向量数据库	Chroma	本地嵌入式，轻量易部署
Web 界面	Streamlit	极简、易上手
开发语言	Python 3.12	搭配 OpenAI SDK 兼容接口
🚀 快速开始
1. 安装依赖
bash
pip install openai chromadb langchain-text-splitters streamlit
2. 配置 API Key（硅基流动）
cmd
set SILICONFLOW_API_KEY=你的硅基流动密钥
3. 准备文档
把你的 .txt 或 .md 文件放进 data/ 文件夹。

4. 建库（离线）
bash
python -m src.ingest
5. 启动网页
bash
streamlit run src/app.py
💡 一键启动
直接双击 start.bat，自动完成环境激活、Key 设置和菜单选择。

📊 评估结果
指标	结果
测试集大小	4 个问题
Top-K	4
Recall@4	100.0%
测试集位于 eval/questions.json，脚本为 eval/evaluate.py。

📁 项目结构
text
mini-rag-agent/
├── data/                # 原始文档
├── index/               # Chroma 向量库持久化目录
├── src/
│   ├── config.py        # API 配置
│   ├── ingest.py        # 离线建库
│   ├── retriever.py     # 检索封装
│   ├── tools.py         # 工具定义
│   ├── memory.py        # 多轮对话记忆
│   ├── agent.py         # Agent 主循环
│   └── app.py           # Streamlit 网页界面
├── eval/
│   ├── questions.json   # 测试集
│   └── evaluate.py      # 评估脚本
├── start.bat            # 一键启动菜单
└── README.md
💡 关键设计
幂等 ID：切分后的 chunk 使用 source::index 作为 ID，重复建库不产生重复向量。

批处理 Embedding：调用硅基流动接口时按 32 条分批，防止单次请求超限。

上下文截断：工具返回结果截断至 4000 字符，防止上下文爆炸。

记忆压缩：超过 8 轮对话丢弃最早的，防止长对话污染上下文。