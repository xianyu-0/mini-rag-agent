import streamlit as st
import sys
import os

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run_agent_stream
from src.memory import Memory

st.set_page_config(page_title="我的知识库 Agent", page_icon="📚")
st.title("📚 个人知识库 Agent")

# 初始化状态
if "memory" not in st.session_state:
    st.session_state.memory = Memory()
if "chat" not in st.session_state:
    st.session_state.chat = []

# 渲染历史对话
for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.write(content)

# 处理新输入
if prompt := st.chat_input("问点什么..."):
    # 显示用户消息
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # 流式显示 AI 回复
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in run_agent_stream(prompt, st.session_state.memory):
            full_response += chunk
            # 末尾加光标，模仿 ChatGPT
            placeholder.markdown(full_response + "▌")
        # 移除光标
        placeholder.markdown(full_response)

    st.session_state.chat.append(("assistant", full_response))