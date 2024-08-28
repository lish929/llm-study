# -*- coding: utf-8 -*-
# @Time    : 2024/8/27/027 18:44
# @Author  : Shining
# @File    : vector_retrieval.py
# @Description :


from dataset import vectordb

question = "什么是民法典？"

# sim_docs = vectordb.similarity_search(question,k=3)
mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
print(mmr_docs)
print(len(mmr_docs))