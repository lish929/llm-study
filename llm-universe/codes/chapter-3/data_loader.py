# -*- coding: utf-8 -*-
# @Time    : 2024/8/26/026 17:37
# @Author  : Shining
# @File    : data_loader.py
# @Description :

"""
PDF
"""

from langchain.document_loaders.pdf import PyMuPDFLoader

pdf_loader = PyMuPDFLoader("中华人民共和国民法典.pdf")
pdf_pages = pdf_loader.load()

pdf_page = pdf_pages[20]

import re

# pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pattern = re.compile(r'[$\u4e00-\u9fff](\n)', re.DOTALL)
print(pdf_page.page_content)
print(repr(pdf_page.page_content))
pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)

print(type(pdf_page.page_content))
print(repr(pdf_page.page_content))
print(pdf_page.page_content)

# print(f"每一个元素的类型：{type(pdf_page)}.",
#     f"该文档的描述性数据：{pdf_page.metadata}",
#     f"查看该文档的内容:\n{pdf_page.page_content}",
#     sep="\n------\n")
