# -*- coding: utf-8 -*-
# @Time    : 2024/8/27/027 17:13
# @Author  : Shining
# @File    : splitter.py
# @Description :


from langchain.text_splitter import RecursiveCharacterTextSplitter

from data_loader import pdf_pages

# 知识库单段文字长度
CHUNK_SIZE = 100
# 知识库相邻文本重合长度
OVERLAP_SIZE = 10

# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    # separators=["\u7b2c"],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)


# split_docs = text_splitter.split_documents(pdf_pages)
split_docs = text_splitter.split_text(pdf_pages[20].page_content[0:500])
print(len(split_docs))
print(split_docs)