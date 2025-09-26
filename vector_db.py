import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import pickle

# --- 1. 환경 설정 및 API 키 로드 ---

# .env 파일에서 환경 변수를 로드합니다.
# 이 함수는 .env 파일에 저장된 키-값 쌍을 시스템 환경 변수처럼 쓸 수 있게 해줍니다.
load_dotenv()

# OpenAI 클라이언트를 초기화합니다.
# os.getenv("OPENAI_API_KEY")를 통해 .env 파일에 저장된 API 키를 가져옵니다.
# 이 클라이언트 객체를 통해 OpenAI의 여러 기능(임베딩 생성 등)을 사용할 수 있습니다.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- 2. 임베딩 생성 함수 ---

def create_embeddings(texts, model="text-embedding-3-small"):
    """
    주어진 텍스트 목록을 OpenAI 임베딩 모델을 사용해 벡터로 변환합니다.

    Args:
        texts (list): 벡터로 변환할 문자열이 담긴 리스트입니다.
                       예: ["첫 번째 문서", "두 번째 문서"]
        model (str): 사용할 OpenAI 임베딩 모델의 이름입니다.
                     "text-embedding-3-small"은 비용과 성능 면에서 효율적입니다.

    Returns:
        np.ndarray: 각 텍스트에 대한 임베딩 벡터가 담긴 NumPy 배열입니다.
                    배열의 각 행이 하나의 텍스트에 대한 벡터에 해당합니다.
    """
    # OpenAI 임베딩 API를 호출합니다.
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    # API 응답에서 임베딩 데이터만 추출하여 리스트로 만듭니다.
    embeddings = [item.embedding for item in response.data]

    # 리스트를 NumPy 배열로 변환하여 반환합니다.
    # NumPy 배열은 벡터 연산에 효율적입니다.
    return np.array(embeddings).astype('float32')


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

if __name__ == "__main__":
    # RAG 시스템에 넣을 원본 데이터 (간단한 예시)
    # 실제로는 파일에서 읽어오거나, 긴 문서를 여러 조각으로 나눈 데이터가 될 수 있습니다.
    my_documents = [
        "RAG는 검색 증강 생성을 의미합니다.",
        "벡터 데이터베이스는 임베딩 벡터를 저장하는 데 사용됩니다.",
        "OpenAI는 강력한 언어 모델을 제공합니다.",
        "FAISS는 벡터 유사도 검색을 위한 라이브러리입니다.",
        "임베딩은 텍스트를 숫자 벡터로 표현하는 것입니다."
    ]

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