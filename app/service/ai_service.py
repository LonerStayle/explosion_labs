import os, pickle, faiss

from app.dto.HelpChatReq import HelpChatReq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.output_parsers import OutputFixingParser
from app.service.parser import combination_parser
from app.dto.CombinationReq import CombinationReq
from app.dto.CombinationRes import CombinationRes
from app.service.prompts import (
    combination_prompt,
    help_chat_prompt,
)
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")


class AiService:
    def __init__(self):
        with open("documents.pkl", "rb") as f:
            raw_docs = pickle.load(f)

        documents = []
        for d in raw_docs:
            if isinstance(d, dict):
                documents.append(
                    Document(page_content=d.get("description", ""), metadata=d)
                )
            else:
                documents.append(d)

        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        faiss_index = faiss.read_index("my_faiss.index")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.vectordb = FAISS(
            embedding_function=self.embeddings,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(len(documents))},
        )
        self.retriver = self.vectordb.as_retriever(search_kwargs={"k": 2})

        self.nano_llm = ChatOpenAI(
            openai_api_key=key, model="gpt-4.1-nano", streaming=True
        )

        self.mini_llm = ChatOpenAI(
            openai_api_key=key, model="gpt-4o-mini", streaming=True
        )
        
        self.fixing_combi_parser = OutputFixingParser.from_llm(
            llm=self.nano_llm,
            parser=combination_parser,
        )

    async def combination_message(self, req: CombinationReq):

        chain = combination_prompt | self.nano_llm | self.fixing_combi_parser
        resp: CombinationRes = await chain.ainvoke(
            {
                "material_a": req.material_a.value,
                "material_b": req.material_b.value,
                "scenario": req.scenario.value,
                "format_instructions": self.fixing_combi_parser.get_format_instructions(),
            }
        )
        return resp

    async def help_message(self, req: HelpChatReq):
        docs = self.retriver.get_relevant_documents(req.question)
        context = "\n".join([d.page_content for d in docs])
        print(context)
        
        chain = help_chat_prompt | self.nano_llm
        resp = await chain.ainvoke(
            {
                "material": req.select_material,
                "question": req.question,
                "context": context,
                "scenario": req.scenario,
            }
        )

        return resp.content
