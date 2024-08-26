# -*- coding: utf-8 -*-
# @Time    : 2024/8/26/026 17:04
# @Author  : Shining
# @File    : openai_api.py
# @Description :

import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


# 读取本地/项目的环境变量。
_ = load_dotenv(find_dotenv())

# 设置代理
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

def openai_embedding(text: str, model: str=None):
    # OPENAI_API_KEY
    api_key = os.environ['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key,base_url="https://api.f2gpt.com")

    # 三种模式：'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
    if model == None:
        model="text-embedding-ada-002"

    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response


if __name__ == '__main__':
    response = openai_embedding(text='要生成 embedding 的输入文本，字符串形式。')
    print(response)