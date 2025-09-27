import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import pickle
import json
from typing import List, Dict

# --- 1. 환경 설정 및 API 키 로드 ---

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# OpenAI 클라이언트를 초기화합니다.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- 2. 임베딩 생성 함수 (이전과 동일) ---

def create_embeddings(docs: List[Dict], model="text-embedding-3-small") -> np.ndarray:
    """
    주어진 문서 목록(docs)을 OpenAI 임베딩 모델을 사용해 벡터로 변환합니다.
    문서의 구조(help_chat용인지, combination용인지)를 자동으로 감지하여 처리합니다.

    [파라미터]
    - docs (List[Dict]): 파일에서 읽어온 딕셔너리(JSON 객체)의 리스트입니다.
    - model (str): 사용할 OpenAI 임베딩 모델의 이름입니다.

    [반환]
    - np.ndarray: 각 문서를 벡터로 변환한 NumPy 배열. FAISS에 넣을 수 있는 형태입니다.
    """
    texts = []
    if not docs:
        return np.array([], dtype="float32")

    first_doc = docs[0]
    if "reactant1" in first_doc:
        print("INFO: 'help_chat_doc' 형식의 데이터를 처리합니다.")
        for d in docs:
            parts = [str(v) for v in d.values() if v]
            texts.append(" | ".join(parts))
    elif "scenario" in first_doc:
        print("INFO: 'combination_doc' 형식의 데이터를 처리합니다.")
        for d in docs:
            parts = [str(v) for v in d.values() if v]
            texts.append(" | ".join(parts))
    else:
        raise ValueError("알 수 없는 문서 형식입니다. 'reactant1' 또는 'scenario' 필드를 확인해주세요.")

    response = client.embeddings.create(
        model=model,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype="float32")

# --- 3. 파일 로드 함수 (오류 해결을 위해 수정됨) ---

def load_data_from_file(path: str) -> List[Dict]:
    """
    [수정된 함수]
    주어진 경로의 파일을 읽어 딕셔너리의 리스트로 반환합니다.
    파일 내용이 '['로 시작하면 표준 JSON 배열로, 그렇지 않으면 JSONL 형식으로 자동 인식하여 처리합니다.
    이것이 JSONDecodeError를 해결하는 핵심입니다.

    [파라미터]
    - path (str): 읽어올 JSON 또는 JSONL 파일의 경로입니다.

    [반환]
    - List[Dict]: 파일 내용을 파싱한 딕셔너리가 담긴 리스트입니다.
    """
    with open(path, "r", encoding="utf-8") as f:
        # 파일의 첫 글자를 읽어 형식을 판단합니다.
        first_char = f.read(1)
        f.seek(0) # 파일 포인터를 다시 처음으로 돌려놓습니다.

        if first_char == '[':
            # 파일이 '['로 시작하면, 파일 전체를 하나의 JSON 배열로 간주하고 읽습니다.
            print(f"INFO: '{path}'를 표준 JSON 배열 파일로 읽습니다.")
            return json.load(f)
        else:
            # 그렇지 않으면, 한 줄에 하나의 JSON이 있는 JSONL 파일로 간주하고 읽습니다.
            print(f"INFO: '{path}'를 JSONL 파일로 읽습니다.")
            docs = []
            for line in f:
                if line.strip(): # 빈 줄은 건너뜁니다.
                    docs.append(json.loads(line))
            return docs

# --- 4. DB 생성 및 저장 로직을 하나의 함수로 묶기 (이전과 동일) ---

def process_and_save_db(source_path: str, output_index_path: str, output_pkl_path: str):
    """
    하나의 소스 파일(JSONL)을 읽어 벡터 DB(.index)와 원본 문서(.pkl)를 생성하고 저장합니다.
    이 함수는 반복적인 작업을 줄이고 코드를 더 명확하게 만들어 유지보수를 쉽게 합니다.

    [파라미터]
    - source_path (str): 처리할 원본 데이터 파일 경로 (예: 'app/data/help_chat_doc.jsonl')
    - output_index_path (str): 생성될 FAISS 인덱스 파일이 저장될 경로 (예: 'help_chat.index')
    - output_pkl_path (str): 원본 문서 목록이 저장될 Pickle 파일 경로 (예: 'help_chat_documents.pkl')
    """
    print(f"\n===== '{source_path}' 파일 처리 시작 =====")

    # 1. 원본 데이터 파일을 로드합니다. (수정된 함수 호출)
    documents = load_data_from_file(source_path)
    if not documents:
        print(f"'{source_path}' 파일에 내용이 없어 건너뜁니다.")
        return

    # 2. OpenAI API를 사용하여 문서를 임베딩 벡터로 변환합니다.
    print("임베딩을 생성합니다...")
    doc_embeddings = create_embeddings(documents)

    # 3. FAISS 인덱스를 생성합니다.
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    # 4. 생성된 인덱스와 원본 문서를 파일로 저장합니다.
    faiss.write_index(index, output_index_path)
    print(f"FAISS 인덱스를 '{output_index_path}' 파일로 저장했습니다. (총 {index.ntotal}개 벡터)")

    with open(output_pkl_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"원본 문서를 '{output_pkl_path}' 파일로 저장했습니다.")
    print("=" * 40)


# --- 5. 메인 실행 부분 (이전과 동일) ---
if __name__ == "__main__":
    
    # 처리할 파일 대상 1: 도움말 챗봇 데이터
    process_and_save_db(
        source_path="app/data/help_chat_doc.jsonl",
        output_index_path="help_chat.index",
        output_pkl_path="help_chat_documents.pkl"
    )

    # 처리할 파일 대상 2: 화학물질 조합 데이터
    process_and_save_db(
        source_path="app/data/combination_doc.jsonl",
        output_index_path="combination.index",
        output_pkl_path="combination_documents.pkl"
    )

    print("\n--- 모든 작업 완료 ---")