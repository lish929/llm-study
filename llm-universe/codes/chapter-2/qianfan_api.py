# -*- coding: utf-8 -*-
# @Time    : 2024/8/23 15:32
# @Author  : Shining
# @File    : qianfan_api.py
# @Description :

# https://console.bce.baidu.com/qianfan/overview

from dotenv import find_dotenv,load_dotenv

_ = load_dotenv(find_dotenv())

import qianfan

def gen_qianfan_messages(prompt):
    """
    :param prompt: 输入
    :return: 请求参数
    """
    messages = [{"role": "user", "content": prompt}]
    return messages

def get_qianfan_completion(prompt,model="Yi-34B-Chat",temperature=0.01):
    """
    :param prompt:
    :param model:
    :param temperature:
    :return:
    """

    chat_completion = qianfan.ChatCompletion()
    messages = gen_qianfan_messages(prompt)
    response = chat_completion.do(
        messages=messages,
        model=model,
        temperature=temperature,
        system="你是一名乐观开朗的个人助理-小李"
    )
    return response["result"]

if __name__ == '__main__':
    result = get_qianfan_completion("你好！请介绍一下自己！")
    print(result)
