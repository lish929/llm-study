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
print(type(pdf_pages))
exit()

for i in range(len(pdf_pages)):

    pdf_page = pdf_pages[i]

    import re

    """
    查看原本数据\n的分布情况:
        1.每一页pdf在读取时，在开头存在页码
        2.保留第一节 第一条 第字前面的换行符
    """
    # print(pdf_page.page_content)
    # print(repr(pdf_page.page_content))

    pattern = re.compile(r'[$\d]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)
    pattern = re.compile(r'[$\d]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)
    pattern = re.compile(r'[$\d]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)

    pattern = re.compile(r'(\n)[$\u4e00-\u7b2b,\u7b2d-\u9fff]', re.DOTALL)
    pdf_page.page_content = re.sub(pattern, "", pdf_page.page_content)
    # print(pdf_page.page_content)
    # print(repr(pdf_page.page_content))


    # print(type(pdf_page.page_content))
    # print(repr(pdf_page.page_content))
    # print(pdf_page.page_content)

    # print(f"每一个元素的类型：{type(pdf_page)}.",
    #     f"该文档的描述性数据：{pdf_page.metadata}",
    #     f"查看该文档的内容:\n{pdf_page.page_content}",
    #     sep="\n------\n")
