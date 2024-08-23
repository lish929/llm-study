# -*- coding: utf-8 -*-
# @Time    : 2024/8/23 16:35
# @Author  : Shining
# @File    : spark_api.py
# @Description :


from dotenv import find_dotenv,load_dotenv

_ = load_dotenv(find_dotenv())

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

