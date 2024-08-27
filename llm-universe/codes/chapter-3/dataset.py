# -*- coding: utf-8 -*-
# @Time    : 2024/8/27/027 17:45
# @Author  : Shining
# @File    : dataset.py
# @Description :


import os
from dotenv import find_dotenv,load_dotenv
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# 配置环境变量
_ = load_dotenv(find_dotenv())
# 读取需要加入数据库的文件 这里只有一个 直接给了
file_paths = ["中华人民共和国民法典.pdf"]

# 创建数据加载器
def create_loader(file_paths):
    loaders = []
    for file_path in file_paths:
        if file_path.endswith("pdf"):
            loaders.append(PyMuPDFLoader(file_path))
        elif file_path.endswith("md"):
            loaders.append(UnstructuredMarkdownLoader(file_path))
    return loaders

# 加载数据
def load_text(loaders):
    texts = []
    for loader in loaders:
        texts.extend(loader.load())
    return texts

# 清洗数据
def wash_data(docs):
    # for i in range(len(docs)):
    pdf_page = docs
    """
    查看原本数据\n的分布情况:
        1.每一页pdf在读取时，在开头存在页码
        2.保留第一节 第一条 第字前面的换行符
    """
    pattern = re.compile(r'[$\d]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)
    pattern = re.compile(r'[$\d]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)
    pattern = re.compile(r'[$\d]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)
    pattern = re.compile(r'(\n)[$\u4e00-\u7b2b,\u7b2d-\u9fff]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)


# 使用 OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 使用百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使

vectordb = None

# if __name__ == '__main__':
loaders = create_loader(file_paths)
# langchain_core.documents.base.Document类型
texts = load_text(loaders)
# 清洗数据
for doc in texts:
    wash_data(doc)
# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
split_docs = text_splitter.split_documents(texts)

from zhipu_embedding import ZhipuAIEmbeddings

zhipu_embedding = ZhipuAIEmbeddings()
persist_directory = r'C:\llm_study\lll-universe\codes\chapter-3'

from langchain.vectorstores.chroma import Chroma

vectordb = Chroma.from_documents(
    documents=split_docs[:20],  # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
    embedding=zhipu_embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")
