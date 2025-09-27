import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import pickle
import json
from typing import List, Any

# --- 1. 환경 설정 및 API 키 로드 ---

# .env 파일에서 환경 변수를 로드합니다.
# 이 함수는 .env 파일에 저장된 키-값 쌍을 시스템 환경 변수처럼 쓸 수 있게 해줍니다.
load_dotenv()

# OpenAI 클라이언트를 초기화합니다.
# os.getenv("OPENAI_API_KEY")를 통해 .env 파일에 저장된 API 키를 가져옵니다.
# 이 클라이언트 객체를 통해 OpenAI의 여러 기능(임베딩 생성 등)을 사용할 수 있습니다.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- 2. 임베딩 생성 함수 ---

def create_embeddings(docs, model="text-embedding-3-small"):
    """
    docs: list of dict (RAG 문서)
    """
    # 문자열 리스트로 변환
    texts = []
    for d in docs:
        parts = []
        if d.get("reactant1"):
            parts.append(d["reactant1"])
        if d.get("reactant2"):
            parts.append(d["reactant2"])
        if d.get("reactant3_catalyst"):
            parts.append(d["reactant3_catalyst"])
        if d.get("products"):
            parts.append(d["products"])
        if d.get("description"):
            parts.append(d["description"])
        if d.get("usage"):
            parts.append(d["usage"])
        texts.append(" | ".join(parts))  # 하나의 문자열로 합침

    # OpenAI 임베딩 생성
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype="float32")  # ✅ numpy 배열로 변환

# --- 3. 벡터 DB (FAISS 인덱스) 생성 및 저장 함수 ---

def build_and_save_vector_db(embeddings, index_path="my_faiss.index"):
    """
    주어진 임베딩 벡터들로부터 FAISS 인덱스를 만들고 파일로 저장합니다.

    Args:
        embeddings (np.ndarray): FAISS 인덱스에 추가할 임베딩 벡터들의 NumPy 배열입니다.
                                 create_embeddings 함수로부터 반환된 값입니다.
        index_path (str): 생성된 FAISS 인덱스를 저장할 파일의 경로입니다.
    """
    # 임베딩 벡터의 차원 수를 확인합니다. (예: "text-embedding-3-small"은 1536 차원)
    dimension = embeddings.shape[1]

    # 가장 기본적인 FAISS 인덱스인 IndexFlatL2를 생성합니다.
    # IndexFlatL2는 모든 벡터를 그대로 저장하고, 검색 시 모든 벡터와 거리를 계산하여
    # 가장 가까운 것을 찾는 가장 간단하고 정확한 방식입니다. (데이터가 아주 많지 않을 때 적합)
    index = faiss.IndexFlatL2(dimension)

    # 생성된 인덱스에 임베딩 벡터들을 추가합니다.
    index.add(embeddings)

    # 완성된 인덱스를 지정된 경로에 파일로 저장합니다.
    faiss.write_index(index, index_path)
    print(f"FAISS 인덱스를 '{index_path}' 파일로 저장했습니다.")
    print(f"총 {index.ntotal}개의 벡터가 DB에 저장되었습니다.")


def save_documents(documents, path="documents.pkl"):
    """
    검색 결과에서 원본 텍스트를 찾아주기 위해, 원본 문서 목록을 별도 파일로 저장합니다.

    Args:
        documents (list): 저장할 원본 텍스트 데이터 목록입니다.
        path (str): 문서 목록을 저장할 파일 경로입니다.
    """
    # pickle을 사용하여 파이썬 리스트 객체를 파일에 그대로 저장합니다.
    with open(path, "wb") as f:
        pickle.dump(documents, f)
    print(f"원본 문서를 '{path}' 파일로 저장했습니다.")


# --- 4. 메인 실행 부분 ---


def load_json_or_jsonl(path: str) -> List[Any]:
    """
    - JSON array (starts with '[') 이면 전체를 json.load로 읽어 리스트 반환
    - 아니면 각 라인을 json.loads로 파싱(빈줄 무시)하여 리스트 반환
    """
    with open(path, "r", encoding="utf-8") as f:
        # 파일의 앞부분에서 첫 의미 있는 문자 하나 읽어 확인
        # (파일이 크지 않다면 전체를 읽어도 되지만, 여기선 앞부분만 본다)
        first_chunk = f.read(2048)
        if not first_chunk:
            return []
        stripped = first_chunk.lstrip()
        if stripped.startswith('['):
            # 파일 전체를 처음부터 읽어 json array 로 처리
            f.seek(0)
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    # array 가 아닌 경우 안전 장치
                    return [data]
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON 배열 파싱 실패: {e}")
        else:
            # JSONL 방식: 첫 부분이 '['가 아니면 라인 단위로 파싱 시도
            f.seek(0)
            docs = []
            bad_lines = []
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    # 문제 있는 라인은 무시하고 기록
                    bad_lines.append((i, line[:200]))
            if bad_lines:
                print("⚠️ 다음 라인들에서 JSON 파싱 실패(무시됨):")
                for ln, snippet in bad_lines:
                    print(f"  line {ln}: {snippet!r}")
            return docs
        
import json
if __name__ == "__main__":
    # RAG 시스템에 넣을 원본 데이터 (간단한 예시)
    # 실제로는 파일에서 읽어오거나, 긴 문서를 여러 조각으로 나눈 데이터가 될 수 있습니다.
    
    my_documents = load_json_or_jsonl(r"C:\PythonProject\explosion_labs\app\data\help_chat_doc.jsonl")
    

    print("1. OpenAI API를 사용하여 임베딩을 생성합니다...")
    # 문서들을 벡터로 변환합니다.
    doc_embeddings = create_embeddings(my_documents)

    print("\n2. FAISS 벡터 DB를 생성하고 저장합니다...")
    # 변환된 벡터로 벡터 DB를 구축하고 파일로 저장합니다.
    build_and_save_vector_db(doc_embeddings, "my_faiss.index")

    # 나중에 벡터 인덱스와 매칭시키기 위해 원본 문서도 저장합니다.
    save_documents(my_documents, "documents.pkl")

    print("\n--- 작업 완료 ---")
    print("이제 'my_faiss.index'와 'documents.pkl' 파일을 RAG 검색 파트에서 사용할 수 있습니다.")