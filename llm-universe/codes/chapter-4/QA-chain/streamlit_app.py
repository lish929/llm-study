# -*- coding: utf-8 -*-
# @Time    : 2024/8/28/028 18:22
# @Author  : Shining
# @File    : streamlit_app.py
# @Description :


import streamlit as st

from llm import ZhipuAILLM

# 设置应用程序标题
st.title("基于LangChain的RAG问答系统")

# 添加文本框 输入ZHIPUAI_API_KEY
ZHIPUAI_API_KEY = st.sidebar.text_input("ZHIPUAI_API_KEY",type="password")

# 身份验证 提示llm由输入响应的内容
def gen_response(inout_text):
    llm = ZhipuAILLM(temperature=0.01,api_key=ZHIPUAI_API_KEY)
    st.info(llm(inout_text))

with st.form('form'):
    text = st.text_area('输入文本:')
    submitted = st.form_submit_button('Submit')
    if submitted:
        gen_response(text)