# -*- coding: utf-8 -*-
# @Time    : 2024/8/27/027 18:22
# @Author  : Shining
# @File    : zhipu_embedding.py
# @Description :


from langchain.embeddings.base import Embeddings
from typing import List

class ZhipuAIEmbeddings(Embeddings):
    def __init__(self):
        self.client = self._create_client()
    def _create_client(self):
        from zhipuai import ZhipuAI
        return ZhipuAI()

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.client.embeddings.create(
            model="embedding-2",
            input=text
        )
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
