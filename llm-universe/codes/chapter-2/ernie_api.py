# -*- coding: utf-8 -*-
# @Time    : 2024/8/23 16:07
# @Author  : Shining
# @File    : ernie_api.py
# @Description :

from dotenv import find_dotenv,load_dotenv

_ = load_dotenv(find_dotenv())

import erniebot
import os

erniebot.api_type = "aistudio"
erniebot.access_token = os.environ.get("EB_ACCESS_TOKEN")

def gen_wenxin_messages(prompt):
    """
    :param prompt: 输入
    :return: 请求参数
    """
    messages = [{"role": "user", "content": prompt}]
    return messages

def get_wenxin_completion(prompt,model="ernie-3.5",temperature=0.01):
    """
    :param prompt:
    :param model:
    :param temperature:
    :return:
    """

    chat_completion = erniebot.ChatCompletion()
    messages = gen_wenxin_messages(prompt)
    response = chat_completion.create(
        messages=messages,
        model=model,
        temperature=temperature,
        system="你是一名乐观开朗的个人助理-小李"
    )
    return response["result"]

if __name__ == '__main__':
    result = get_wenxin_completion("你好！请介绍一下自己！")
    print(result)
