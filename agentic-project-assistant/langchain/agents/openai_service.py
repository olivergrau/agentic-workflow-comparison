from __future__ import annotations

"""Wrapper around LangChain's OpenAI utilities used by the agents."""

from typing import Any, List, Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class OpenAIService:
    """Service class that proxies chat and embedding calls via LangChain."""

    def __init__(self, api_key: str, base_url: str = "https://openai.vocareum.com/v1") -> None:
        self.chat_model = ChatOpenAI(api_key=api_key, base_url=base_url)
        self.embed_model = OpenAIEmbeddings(api_key=api_key, base_url=base_url)

    def _convert_messages(self, messages: List[Dict[str, str]]):
        lc_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                lc_messages.append(AIMessage(content=content))
        return lc_messages

    def chat(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature", 0)
        lc_messages = self._convert_messages(messages)
        response = self.chat_model.invoke(lc_messages, temperature=temperature)

        class Choice:  # mimic OpenAI SDK response structure
            def __init__(self, message):
                self.message = type("Message", (), {"content": message.content})

        class Response:
            def __init__(self, message):
                self.choices = [Choice(message)]

        return Response(response)

    def embed(self, **kwargs: Any) -> Any:
        text = kwargs.get("input")
        embedding = self.embed_model.embed_query(text)

        class Data:
            def __init__(self, embedding):
                self.embedding = embedding

        class Response:
            def __init__(self, embedding):
                self.data = [Data(embedding)]

        return Response(embedding)
