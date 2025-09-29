import os
import pickle
import faiss
from rank_bm25 import BM25Okapi
import numpy as np
from app.service.contract import is_chemical_question, compute_hint_case_advanced
from app.common.GameScenario import GameScenario
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.output_parsers import OutputFixingParser

# --- DTO 및 서비스 모듈 임포트 ---
from app.dto.CombinationReq import CombinationReq
from app.dto.HelpChatReq import HelpChatReq
from app.service.parser import combination_parser
from app.dto.CombinationRes import CombinationRes
from app.service.prompts import (
    comment_prompt,
    combination_prompt,
    help_chat_prompt,
)

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()

def normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.ones_like(scores)  # 전부 같은 점수면 1로 세팅
    return (scores - scores.min()) / (scores.max() - scores.min())
    

class AiService:
    def __init__(self):

        # --- 더 안정적인 파일 경로 설정 ---
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        combination_index_path = os.path.join(project_root, "combination.index")
        combination_pkl_path = os.path.join(project_root, "combination_documents.pkl")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # --- 1. 도움말(Help Chat)용 벡터 DB 및 검색기(Retriever) 설정 ---
        help_chat_index = faiss.read_index(combination_index_path)
    
        with open(combination_pkl_path, "rb") as f:
            raw_help_docs = pickle.load(f)
        combination_documents = []
        for d in raw_help_docs:
            document_text = f"{d.get('scenario')} | {d.get('material_a')} | {d.get('material_b')}"
            combination_documents.append(
                Document(page_content=document_text, metadata=d)
            )


        help_chat_docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(combination_documents)})
        self.help_vectordb = FAISS(
            embedding_function=self.embeddings,
            index=help_chat_index,
            docstore=help_chat_docstore,
            index_to_docstore_id={i: str(i) for i in range(len(combination_documents))},
        )
        
        self.help_retriver = self.help_vectordb.as_retriever(search_kwargs={"k": 1})
        self.help_chat_documents = combination_documents
        tokenized_corpus = [doc.page_content.split() for doc in combination_documents]
        self.help_bm25 = BM25Okapi(tokenized_corpus)


        # --- 2. 조합(Combination)용 벡터 DB 및 검색기(Retriever) 설정 ---
        combination_index = faiss.read_index(combination_index_path)
        with open(combination_pkl_path, "rb") as f:
            raw_combination_docs = pickle.load(f)

        combination_documents = []
        for d in raw_combination_docs:
            document_text = f"{d.get('scenario')} | {d.get('material_a')} | {d.get('material_b')}"
            combination_documents.append(
                Document(page_content=document_text, metadata=d)
            )

        combination_docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(combination_documents)})
        
        self.combination_vectordb = FAISS(
            embedding_function=self.embeddings,
            index=combination_index,
            docstore=combination_docstore,
            index_to_docstore_id={i: str(i) for i in range(len(combination_documents))},
        )
        self.combination_retriever = self.combination_vectordb.as_retriever(search_kwargs={"k": 1})

        # --- 3. LLM (언어 모델) 및 파서(Parser) 설정 ---
        self.mini_llm = ChatOpenAI(
             model="gpt-4o-mini"
        )

        self.nano_llm = ChatOpenAI(
             model="gpt-4.1-nano"
        )

        self.fixing_combi_parser = OutputFixingParser.from_llm(
            llm=self.mini_llm,
            parser=combination_parser,
        )

    
    
    async def combination_message(self, req: CombinationReq):
        query = f"{req.scenario.value} | {req.material_a.value} | {req.material_b.value}"
        docs = self.combination_retriever.get_relevant_documents(query)

        if not docs:
            context = "일치하는 조합 데이터를 찾지 못했습니다."
        else:
            context = str(docs[0].metadata)
        
        print("문서 내용:",context)
        chain = combination_prompt | self.mini_llm | self.fixing_combi_parser
        resp = await chain.ainvoke({
            "material_a": req.material_a.value,
            "material_b": req.material_b.value,
            "scenario": req.scenario.value,
            "context": context,
            "format_instructions": self.fixing_combi_parser.get_format_instructions(),
        })

        if (resp.result_1 == resp.scenario_answer
            or resp.result_2 == resp.scenario_answer):
            resp.result_state = "SUCCESS"
        elif resp.result_1 or resp.result_2:
            resp.result_state = "BAD"
        else:
            resp.result_state = "NOTHING"
        
        comment_chain = comment_prompt | self.nano_llm
        comment = await comment_chain.ainvoke({"result_state": resp.result_state})
        resp.comment = comment.content
        return resp


    async def help_message(self, req: HelpChatReq):
        hint_case = compute_hint_case_advanced(req.scenario, req.select_material, req.question)
        if not is_chemical_question(req.question):
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