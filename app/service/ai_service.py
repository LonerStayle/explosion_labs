import os
from app.dto import CombinationReq
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
class AiService:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=key,
            model="gpt-4o-mini",
            streaming=True
        )

    async def combination_message(self,  req: CombinationReq):
        messages = [
            SystemMessage(content="너는 게임 속 AI 조합 가이드야. 짧고 간결하게 답해."),
            HumanMessage(content="test")
        ]

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content