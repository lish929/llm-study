# -*- coding: utf-8 -*-
# @Time    : 2024/8/28/028 9:57
# @Author  : Shining
# @File    : ChatGPT.py
# @Description :


from dotenv import find_dotenv,load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

# 获取api key
_ = load_dotenv(find_dotenv())
open_api_key = os.environ.get("OPENAI_API_KEY")


"""
model="gpt-4o",
temperature=0,
max_tokens=None,
timeout=None,
max_retries=2,
# api_key="...",
# base_url="...",
# organization="...",
# other params...
"""
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0,api_key=open_api_key,base_url="https://api.f2gpt.com")

# output = llm.invoke("请介绍一下自己！")
# print(output)

system_template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template)
])

text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
messages = chat_prompt.format_messages(input_language="中文", output_language="英文", text=text)
output = llm.invoke(messages)
print(output)