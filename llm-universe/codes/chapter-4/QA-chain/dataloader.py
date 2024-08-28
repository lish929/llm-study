# -*- coding: utf-8 -*-
# @Time    : 2024/8/28/028 14:40
# @Author  : Shining
# @File    : dataloader.py
# @Description :


from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re

class DataLoader(object):
    def __init__(self,source_root):
        self.source_root = source_root
        self.file_paths = self._get_file_paths()
        self.loaders = self._create_loaders()
        self.docs = self._load_docs()
        self.wash_docs = self._wash_datas()

        # 实际应用中 在不同的任务中 使用不同的分割器至关重要
        self.spliter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                "\u7b2b"
                " ",
                "",
            ],
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False
        )
        self.split_docs = self._split_datas()

    def __call__(self, *args, **kwargs):
        pass

    # 获取所有本地文档路径
    def _get_file_paths(self):
        file_paths = []
        for item in os.scandir(self.source_root):
            file_paths.append(item.path)
        return file_paths

    # 根据文档类型创建数据加载器
    def _create_loaders(self):
        loaders = []
        for file_path in self.file_paths:
            if file_path.endswith("pdf"):
                loaders.append(PyMuPDFLoader(file_path))
            elif file_path.endswith("md"):
                loaders.append(UnstructuredMarkdownLoader(file_path))
        return loaders

    # 加载数据 [<class 'langchain_core.documents.base.Document'>]
    def _load_docs(self):
        docs = []
        for loader in self.loaders:
            docs.extend(loader.load())
        return docs

    # 清洗文档 将\n\n替换为\n
    def _wash_datas(self):
        for doc in self.docs:
            pattern = re.compile(r'(\n\n)', re.DOTALL)
            doc.page_content = re.sub(pattern, "\n", doc.page_content)
        return self.docs
    # 切分文档
    def _split_datas(self):
        split_docs = self.spliter.split_documents(self.docs)
        return split_docs





if __name__ == '__main__':
    dataloader = DataLoader(r"C:\llm_study\lll-universe\codes\chapter-4\QA-chain\resource")
    split_docs = dataloader.split_docs
    for split_doc in split_docs:
        print(split_doc)