# -*- coding: utf-8 -*-
# @Time    : 2024/8/28/028 11:22
# @Author  : Shining
# @File    : spark.py
# @Description :

from dotenv import find_dotenv,load_dotenv
import os

_ = load_dotenv(find_dotenv())
IFLYTEK_SPARKAI_APP_ID = os.environ.get("IFLYTEK_SPARKAI_APP_ID")
IFLYTEK_SPARKAI_API_SECRET = os.environ.get("IFLYTEK_SPARKAI_API_SECRET")
IFLYTEK_SPARKAI_API_KEY = os.environ.get("IFLYTEK_SPARKAI_API_KEY")

def gen_spark_params(model):
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

from langchain_community.llms import SparkLLM

spark_api_url = gen_spark_params(model="v3.5")["spark_url"]
# Load the model(默认使用 v3.0)
llm = SparkLLM(
    spark_api_url=spark_api_url,
    spark_app_id=IFLYTEK_SPARKAI_APP_ID,
    spark_api_key=IFLYTEK_SPARKAI_API_KEY,
    spark_api_secret=IFLYTEK_SPARKAI_API_SECRET
               )  #指定 v1.5版本
res = llm("你好，请你自我介绍一下！")
print(res)