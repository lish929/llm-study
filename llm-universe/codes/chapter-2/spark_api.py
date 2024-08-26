# -*- coding: utf-8 -*-
# @Time    : 2024/8/23 16:35
# @Author  : Shining
# @File    : spark_api.py
# @Description :


from dotenv import find_dotenv,load_dotenv

_ = load_dotenv(find_dotenv())

import os
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

import os
import sparkAPI


def gen_spark_params(model):
    """
    :param model: 使用模型
    :return: 请求参数
    """

    spark_url_tpl = "wss://spark-api.xf-yun.com/{}/chat"
    model_params_dict = {
        # v1.5 版本
        "v1.5": {
            "domain": "general", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v1.1") # 云端环境的服务地址
        },
        # v2.0 版本
        "v2.0": {
            "domain": "generalv2", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v2.1") # 云端环境的服务地址
        },
        # v3.0 版本
        "v3.0": {
            "domain": "generalv3", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.1") # 云端环境的服务地址
        },
        # v3.5 版本
        "v3.5": {
            "domain": "generalv3.5", # 用于配置大模型版本
            "spark_url": spark_url_tpl.format("v3.5") # 云端环境的服务地址
        }
    }
    return model_params_dict[model]

def gen_spark_messages(prompt):
    """
    :param prompt: 提示词
    :return: 请求参数
    """
    messages = [ChatMessage(role="user", content=prompt)]
    return messages


def get_completion(prompt, model="v3.5", temperature=0.1):
    """
    :param prompt: 提示词
    :param model: 模型 默认3.5
    :param temperature: 温度系数
    :return:
    """
    spark_llm = ChatSparkLLM(
        spark_api_url=gen_spark_params(model)["spark_url"],
        spark_app_id=os.environ["SPARKAI_APP_ID"],
        spark_api_key=os.environ["SPARKAI_API_KEY"],
        spark_api_secret=os.environ["SPARKAI_API_SECRET"],
        spark_llm_domain=gen_spark_params(model)["domain"],
        temperature=temperature,
        streaming=False,
    )
    messages = gen_spark_messages(prompt)
    handler = ChunkPrintHandler()
    # 当 streaming设置为 False的时候, callbacks 并不起作用
    response = spark_llm.generate([messages], callbacks=[handler])
    return response

if __name__ == '__main__':
    response = get_completion("你好")
    result = response.generations[0][0].text
    print(result)