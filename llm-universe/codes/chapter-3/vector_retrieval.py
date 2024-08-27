# -*- coding: utf-8 -*-
# @Time    : 2024/8/27/027 18:44
# @Author  : Shining
# @File    : vector_retrieval.py
# @Description :

from langchain.vectorstores import Chroma

# from dataset import vectordb

vectordb = Chroma(persist_directory=r"C:\llm_study\lll-universe\codes\chapter-3")
question = "什么是民法典？"


sim_docs = vectordb.similarity_search(question,k=3)
print(sim_docs)
print(len(sim_docs))