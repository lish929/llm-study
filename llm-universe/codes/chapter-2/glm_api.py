# -*- coding: utf-8 -*-
# @Time    : 2024/8/26/026 15:21
# @Author  : Shining
# @File    : glm_api.py
# @Description :

from dotenv import find_dotenv,load_dotenv

_ = load_dotenv(find_dotenv())

def gen_glm_params(prompt):
    """
    :param prompt: 提示词
    :return: 请求参数
    """
    messages = [{"role": "user", "content": prompt}]
    return messages

import os
from zhipuai import ZhipuAI

client = ZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"]
)


def get_completion(prompt, model="glm-4", temperature=0.95):
    """
    :param prompt: 提示词
    :param model: 模型
    :param temperature: 温度系数
    :return:
    """
    messages = gen_glm_params(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

if __name__ == '__main__':
    result = get_completion("你好！")
    print(result)