# -*- coding: utf-8 -*-
# @Time    : 2024/8/23 14:48
# @Author  : Shining
# @File    : openai_api.py
# @Description :

import os
from dotenv import find_dotenv,load_dotenv

# 读取本地/项目环境变量
_ = load_dotenv(find_dotenv())

# 设置代理
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.f2gpt.com"
)

# completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         # System Prompt
#         {"role": "system", "content": "You are a helpful assistant."},
#         # User Prompt
#         {"role": "user", "content": "Hello!"}
#     ]
# )
#
# print(completion.choices[0].message.content)

def get_gpt_messages(prompt):
    """
    :param prompt: 输入
    :return: 请求参数prompt
    """
    messages = [
        {"role":"user","content":prompt}
    ]
    return messages

def get_gpt_completion(prompt,model="gpt-3.5-turbo",temperature=0):
    """
    :param prompt: 输入
    :param model: 使用模型
    :param temperature: 温度系数
    :return: 模型输出
    """

    response = client.chat.completions.create(
        model=model,
        messages=get_gpt_messages(prompt),
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

print(get_gpt_completion("你好"))