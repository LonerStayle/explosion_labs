import os
from app.dto import CombinationReq
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.service.prompts import COMBINATION_MESSAGE_SYSTEM, HELP_MESSAGE_SYSTEM, combination_prompt
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
class AiService:
    def __init__(self):
        self.mini_llm = ChatOpenAI(
            openai_api_key=key,
            model="gpt-4o-mini",
            streaming=True
        )

        self.nano_llm = ChatOpenAI(
            openai_api_key=key,
            model="gpt-4.1-nano",
            streaming=True
        )

    async def combination_message(self,  req: CombinationReq):
        messages = [
            SystemMessage(content=COMBINATION_MESSAGE_SYSTEM),
            HumanMessage(content="test")
        ]

        combination_prompt

        async for chunk in self.mini_llm.astream(messages):
            if chunk.content:
                yield chunk.content


    async def help_message(self, question):
        messages = [
            SystemMessage(content=HELP_MESSAGE_SYSTEM),
            HumanMessage(content=question)
        ]
        async for chunk in self.nano_llm.astream(messages):
            if chunk.content:
                yield chunk.content

    