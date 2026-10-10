import streamlit as st
import sys
import os

# 把当前路径加入系统路径，确保能引用到 src 里的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run_agent
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
    
    # 显示 AI 回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 注意：这里我们暂时没有传 set 的 key，是用环境变量
            answer = run_agent(prompt, st.session_state.memory)
            st.write(answer)
            st.session_state.chat.append(("assistant", answer))