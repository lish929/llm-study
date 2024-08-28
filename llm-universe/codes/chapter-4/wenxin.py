# -*- coding: utf-8 -*-
# @Time    : 2024/8/28/028 10:48
# @Author  : Shining
# @File    : wenxin.py
# @Description :


from dotenv import find_dotenv, load_dotenv
# from langchain_community.llms import QianfanLLMEndpoint
import os

_ = load_dotenv(find_dotenv())
QIANFAN_AK = os.environ["QIANFAN_AK"]
QIANFAN_SK = os.environ["QIANFAN_SK"]

# llm = QianfanLLMEndpoint(model="Yi-34B-Chat",streaming=True)
# output = llm("你好，请介绍一下自己！")
# print(output)

from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

import qianfan


class WenxinLLM(LLM):
    model: str = "Yi-34B-Chat"
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    # Secret_Key
    secret_key: str = None
    # 系统消息
    system: str = None

    def _call(self,prompt: str,stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,**kwargs: Any,) -> str:
        def gen_wenxin_messages(prompt):
            messages = [{"role": "user", "content": prompt}]
            return messages

        chat_comp = qianfan.ChatCompletion(ak=self.api_key,sk=self.secret_key)
        message = gen_wenxin_messages(prompt)

        resp = chat_comp.do(messages=message,
                            model=self.model,
                            temperature=self.temperature,
                            system=self.system)

        return resp["result"]

    @property
    def _llm_type(self) -> str:
        return "wenxin"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": "CustomChatModel",
        }



llm = WenxinLLM(api_key=QIANFAN_AK, secret_key=QIANFAN_SK, system="你是一个助手！")
output = llm.invoke("你好，请介绍一下自己！")
print(output)