# C:\explosion_labs\app\service\ai_service.py (최종 완성본 + 디버깅 코드)

import os
import pickle
import faiss
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.output_parsers import OutputFixingParser

# --- DTO 및 서비스 모듈 임포트 ---
from app.dto.CombinationReq import CombinationReq
from app.dto.HelpChatReq import HelpChatReq
from app.service.parser import combination_parser
from app.service.prompts import combination_prompt, help_chat_prompt

# --- [핵심 수정] .env 파일에서 API 키를 로드하고 변수에 저장합니다. ---
# 이 코드는 반드시 클래스 바깥, 파일의 최상단에 있어야 합니다.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- [디버깅 코드] API 키가 제대로 로드되었는지 확인합니다. ---
# 스크립트 실행 시 터미널에 "OpenAI API Key Loaded: True" 라고 나와야 정상입니다.
# 만약 "False"라고 나오면 .env 파일에 문제가 있는 것입니다.
print(f"OpenAI API Key Loaded: {OPENAI_API_KEY is not None and OPENAI_API_KEY.startswith('sk-')}")


class AiService:
    def __init__(self):
        """
        AiService 클래스가 처음 생성될 때(서버 시작 시) 한 번만 실행되는 초기화 함수입니다.
        AI 모델, 임베딩 모델, 그리고 RAG에 필요한 벡터 데이터베이스를 미리 로드하여
        API 요청이 왔을 때 빠르게 처리할 수 있도록 준비합니다.
        """
        # --- 더 안정적인 파일 경로 설정 ---
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        help_chat_index_path = os.path.join(project_root, "help_chat.index")
        help_chat_pkl_path = os.path.join(project_root, "help_chat_documents.pkl")
        combination_index_path = os.path.join(project_root, "combination.index")
        combination_pkl_path = os.path.join(project_root, "combination_documents.pkl")

        # --- 공통 설정: 임베딩 모델 로드 ---
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # --- 1. 도움말(Help Chat)용 벡터 DB 및 검색기(Retriever) 설정 ---
        help_chat_index = faiss.read_index(help_chat_index_path)
        
        with open(help_chat_pkl_path, "rb") as f:
            raw_help_docs = pickle.load(f)

        help_chat_documents = []
        for d in raw_help_docs:
            document_text = " | ".join([str(v) for v in d.values() if v])
            help_chat_documents.append(
                Document(page_content=document_text, metadata=d)
            )

        help_chat_docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(help_chat_documents)})
        
        self.help_vectordb = FAISS(
            embedding_function=self.embeddings,
            index=help_chat_index,
            docstore=help_chat_docstore,
            index_to_docstore_id={i: str(i) for i in range(len(help_chat_documents))},
        )
        self.help_retriever = self.help_vectordb.as_retriever(search_kwargs={"k": 2})

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
        # 파일 최상단에서 정의한 OPENAI_API_KEY 변수를 여기서 사용합니다.
        self.nano_llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini"
        )
        self.fixing_combi_parser = OutputFixingParser.from_llm(
            llm=self.nano_llm,
            parser=combination_parser,
        )

    async def combination_message(self, req: CombinationReq):
        query = f"{req.scenario.value} | {req.material_a.value} | {req.material_b.value}"
        docs = self.combination_retriever.get_relevant_documents(query)

        if not docs:
            context = "일치하는 조합 데이터를 찾지 못했습니다."
        else:
            context = str(docs[0].metadata)
        
        chain = combination_prompt | self.nano_llm | self.fixing_combi_parser
        
        resp = await chain.ainvoke({
            "material_a": req.material_a.value,
            "material_b": req.material_b.value,
            "scenario": req.scenario.value,
            "context": context,
            "format_instructions": self.fixing_combi_parser.get_format_instructions(),
        })
        return resp

    async def help_message(self, req: HelpChatReq):
        docs = self.help_retriever.get_relevant_documents(req.question)
        context = "\n".join([d.page_content for d in docs])
        
        chain = help_chat_prompt | self.nano_llm
        resp = await chain.ainvoke({
            "material": req.select_material,
            "question": req.question,
            "context": context,
            "scenario": req.scenario,
        })
        return resp.content