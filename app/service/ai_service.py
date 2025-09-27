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
        faiss_index = faiss.read_index("help_chat.index")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 도움말(help_chat)용 벡터 DB 및 검색기(Retriever) 설정
        # vector_db.py에서 생성한 도움말용 FAISS 인덱스 파일 로드
        
# -------------------------
        with open("help_chat_docments.pkl", "rb") as f:
            raw_docs = pickle.load(f)
        # 불러온 원본 문서를 LangChain에서 사용하기 좋은 Document 객체로 변환
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
        self.help_retriver = self.help_vectordb.as_retriever(search_kwargs={"k": 2})
# -------------------------
# 화인님이 하실 부분 위 헬프챗 도큐먼트 코드 참고 
# Combination용 벡터 DB 및 검색기(Retriever) 설정

# -------------------------
        # vector_db.py에서 생성한 화학물질 조합 데이터용 FAISS 인덱스 파일 로드
        # 조합 데이터용 문서 저장소(docstore)를 설정합니다.
        combination_docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(combination_documents)})
        
        # 조합 데이터용 LangChain FAISS 벡터DB 객체를 생성합니다.
        self.combination_vectordb = FAISS(
            embedding_function=self.embeddings,
            index=combination_index,
            docstore=combination_docstore,
            index_to_docstore_id={i: str(i) for i in range(len(combination_documents))},
        )
        # 조합 검색기는 가장 유사한 1개(k=1)의 결과만 찾으면 충분합니다.
        self.combination_retriever = self.combination_vectordb.as_retriever(search_kwargs={"k": 1})

        # --- 3. LLM (언어 모델) 및 파서(Parser) 설정 ---
        # 상대적으로 간단한 작업에 사용할 가볍고 빠른 모델
        self.nano_llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini"
        )
        # LLM이 JSON 형식을 잘못 출력했을 때, 스스로 수정하도록 돕는 파서입니다.
        self.fixing_combi_parser = OutputFixingParser.from_llm(
            llm=self.nano_llm,
            parser=combination_parser,
        )

    async def combination_message(self, req: CombinationReq):
        """
        화학물질 조합 요청을 받아 RAG를 통해 정확한 결과를 찾아 LLM으로 처리하는 비동기 함수.

        [파라미터]
        - req (CombinationReq): 플레이어가 요청한 조합 재료(material_a, material_b)와
                                현재 시나리오(scenario)가 담긴 데이터 객체.
        """
        # --- [수정] RAG 로직 추가 ---
        # 1. 벡터 DB에서 검색할 질문(Query)을 만듭니다.
        #    요청받은 시나리오와 재료들을 합쳐서 가장 유사한 데이터를 찾도록 합니다.
        query = f"{req.scenario.value} | {req.material_a.value} | {req.material_b.value}"

        # 2. 위에서 만든 검색기(retriever)로 가장 유사한 문서를 DB에서 찾습니다.
        #    .get_relevant_documents()는 LangChain 라이브러리에서 제공하는 검색 함수입니다.
        docs = self.combination_retriever.get_relevant_documents(query)

        # 3. 찾은 문서(docs)를 LLM 프롬프트에 넣어줄 'context' 문자열로 가공합니다.
        #    검색된 문서가 없다면 빈 문자열을, 있다면 찾은 문서의 전체 내용(metadata)을
        #    문자열로 만들어 context로 사용합니다.
        if not docs:
            context = "일치하는 조합 데이터를 찾지 못했습니다."
        else:
            # docs[0]는 검색된 문서 중 가장 유사도가 높은 첫 번째 문서를 의미합니다.
            # .metadata에는 .pkl 파일에 저장했던 원본 딕셔너리 전체가 들어있습니다.
            context = str(docs[0].metadata)
        
        # LangChain의 체인(Chain)을 구성합니다.
        # [프롬프트 템플릿] -> [LLM] -> [출력 파서] 순서로 데이터가 처리됩니다.
        chain = combination_prompt | self.nano_llm | self.fixing_combi_parser
        
        # 4. 체인을 실행(ainvoke)합니다. 프롬프트에 필요한 모든 값을 딕셔너리 형태로 전달합니다.
        resp = await chain.ainvoke({
            "material_a": req.material_a.value,
            "material_b": req.material_b.value,
            "scenario": req.scenario.value,
            "context": context, # RAG를 통해 찾은 '참고 자료'를 여기에 전달합니다.
            "format_instructions": self.fixing_combi_parser.get_format_instructions(),
        })
        return resp

    async def help_message(self, req: HelpChatReq):
        """
        도움말 요청을 받아 RAG를 통해 관련 지식을 찾아 LLM으로 답변을 생성하는 비동기 함수.

        [파라미터]
        - req (HelpChatReq): 플레이어의 질문(question), 선택 재료, 시나리오가 담긴 데이터 객체.
        """
        # 1. 사용자의 질문으로 도움말 DB에서 관련 문서를 찾습니다.
        docs = self.help_retriever.get_relevant_documents(req.question)
        # 2. 찾은 문서들의 핵심 내용(page_content)을 합쳐서 context를 만듭니다.
        context = "\n".join([d.page_content for d in docs])
        
        # 도움말 생성용 체인을 구성하고 실행합니다.
        chain = help_chat_prompt | self.nano_llm
        resp = await chain.ainvoke({
            "material": req.select_material,
            "question": req.question,
            "context": context,
            "scenario": req.scenario,
        })
        # LLM의 답변 내용(.content)을 최종 결과로 반환합니다.
        return resp.conten