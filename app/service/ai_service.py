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
from rank_bm25 import BM25Okapi
import numpy as np
from app.service.contract import is_chemical_question, compute_hint_case_advanced
from app.common.GameScenario import GameScenario
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

def normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.ones_like(scores)  # 전부 같은 점수면 1로 세팅
    return (scores - scores.min()) / (scores.max() - scores.min())
    

class AiService:
    def __init__(self):
        faiss_index = faiss.read_index("my_faiss.index")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# -------------------------
        with open("help_chat_docments.pkl", "rb") as f:
            raw_docs = pickle.load(f)

        help_chat_documents = []
        for d in raw_docs:
            if isinstance(d, dict):
                help_chat_documents.append(
                    Document(page_content=d.get("description", ""), metadata=d)
                )
            else:
                help_chat_documents.append(d)
        

        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(help_chat_documents)})
        self.help_vectordb = FAISS(
            embedding_function=self.embeddings,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(len(help_chat_documents))},
        )
        self.help_retriver = self.help_vectordb.as_retriever(search_kwargs={"k": 1})
        self.help_chat_documents = help_chat_documents

        # --- BM25 ---
        tokenized_corpus = [doc.page_content.split() for doc in help_chat_documents]
        self.help_bm25 = BM25Okapi(tokenized_corpus)

# -------------------------
# 화인님이 하실 부분 위 헬프챗 도큐먼트 코드 참고 
# -------------------------


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
        # ----- 여기다가 RAG 전략 세팅 ------ 
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
        hint_case = compute_hint_case_advanced(req.scenario, req.select_material, req.question)
        if not is_chemical_question(req.question):
            print("0000:", 0000)
            chain = help_chat_prompt | self.nano_llm
            resp = await chain.ainvoke(
                {
                    "material": req.select_material,
                    "question": req.question,
                    "context": "일상 대화중",
                    "hint_case": hint_case, 
                    "scenario": req.scenario,
                }
            )
            return resp.content
        is_use_scenario = (req.scenario == GameScenario.USE_HNO3HCL) or (req.scenario == GameScenario.USE_HNO3HCL)
        if is_use_scenario:
            context = "사용 시나리오 입니다. 그래서 화합물을 제공하지 않습니다."
        else:
            tokenized_query = req.question.split()
            bm25_scores = self.help_bm25.get_scores(tokenized_query)
            bm25_norm = normalize(bm25_scores)

            query_emb = self.embeddings.embed_query(req.question)
            query_emb = np.array([query_emb]).astype("float32")  # (1, dim) 형태로 변환

            D, I = self.help_vectordb.index.search(query_emb, len(self.help_chat_documents))

            faiss_sims = 1 / (1 + D[0])   # distance → similarity 변환
            faiss_norm = normalize(faiss_sims)

            dense = 0.5
            bm = 0.5
            hybrid_scores = dense * faiss_norm + bm * bm25_norm


            # 상위 문서 선택
            top_idx = hybrid_scores.argsort()[::-1][:1]
            final_docs = [self.help_chat_documents[i] for i in top_idx]
            context = "\n".join([d.page_content for d in final_docs])
            print("context 호출:", context)

        # --- LLM 호출 ---
        chain = help_chat_prompt | self.nano_llm
        resp = await chain.ainvoke(
            {
                "material": req.select_material,
                "question": req.question,
                "context": context,
                "scenario": req.scenario,
                "hint_case":hint_case
            }
        )
        return resp.content