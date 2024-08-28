# -*- coding: utf-8 -*-
# @Time    : 2024/8/28/028 17:01
# @Author  : Shining
# @File    : qa_chain.py
# @Description :

from dotenv import find_dotenv,load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
import os

from embedding import ZhipuAIEmbeddings
from llm import ZhipuAILLM

_ = load_dotenv(find_dotenv())
ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY")

# 加载向量数据库
def load_database(database_path=r"C:\llm_study\lll-universe\codes\chapter-4\QA-chain\db",embedding=ZhipuAIEmbeddings()):
    vectordb = Chroma(
        persist_directory=database_path,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

# 创建llm
def create_llm():
    llm = ZhipuAILLM()
    return llm

if __name__ == '__main__':
    vectordb = load_database()
    # question = "什么是民法典？"
    # docs = vectordb.similarity_search(question,k=3)
    # for doc in docs:
    #     print(doc.page_content)
    llm = create_llm()
    # print(llm.invoke("人民法院审理宣告死亡案件时，何人会被定义为利害关系人？"))

    template = '''使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。最多使用三句话。尽量使答案简明扼要。{context}问题: {question}'''
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
    # qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )


    question = "那我应该如何获得民法典的主要内容？"
    result = qa({"question":question})
    print(result["answer"])

