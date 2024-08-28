# -*- coding: utf-8 -*-
# @Time    : 2024/8/28/028 16:25
# @Author  : Shining
# @File    : dattaset.py
# @Description :

from dataloader import DataLoader
from embedding import ZhipuAIEmbeddings

from dotenv import find_dotenv,load_dotenv
from langchain.vectorstores.chroma import Chroma

_ = load_dotenv(find_dotenv())

resource_path = r"C:\llm_study\lll-universe\codes\chapter-4\QA-chain\resource"
database_path = r"C:\llm_study\lll-universe\codes\chapter-4\QA-chain\db"

dataloader = DataLoader(resource_path)
embedding = ZhipuAIEmbeddings()
vectordb = Chroma.from_documents(
    dataloader.split_docs,
    embedding=embedding,
    persist_directory=database_path
)
vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")
