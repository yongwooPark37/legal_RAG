"""
Vector DB
"""

from typing import List, Dict, Optional
from .simple_vector_store import SimpleVectorStore


class LegalVectorStore:
    """
    법률 문서 벡터 스토어
    
    SimpleVectorStore 사용
    Metadata 필터링 지원 (카테고리, 연도 등)
    """
    
    def __init__(self, persist_directory: str = "./simple_db", 
                 collection_name: str = "legal_documents"):
        """
        Args:
            persist_directory: DB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # SimpleVectorStore 사용
        self.collection = SimpleVectorStore(persist_directory=persist_directory)
        print(f"Using SimpleVectorStore (Python 3.14 compatible)")
        if self.collection.get_count() > 0:
            print(f"  Documents: {self.collection.get_count()}")
    
    def add_documents(self, 
                     texts: List[str], 
                     embeddings: List[List[float]],
                     metadatas: List[Dict],
                     ids: List[str]):
        """
        문서를 벡터 DB에 추가
        
        Args:
            texts: 텍스트 리스트
            embeddings: 임베딩 벡터 리스트
            metadatas: 메타데이터 리스트
                예: {'book_id': 'LJU000001', 'category': '민사소송법', 
                     'publication_year': 2015, 'chunk_index': 0}
            ids: 문서 ID 리스트 (고유해야 함)
        """
        # SimpleVectorStore의 add_documents 호출
        self.collection.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, 
              query_embedding: List[float],
              k: int = 10,
              filters: Optional[Dict] = None) -> List[Dict]:
        """
        벡터 검색
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            k: 반환할 문서 수
            filters: 메타데이터 필터
                예: {'category': '민사소송법'}
                예: {'publication_year': {'$gte': 2015}}
        
        Returns:
            검색 결과 리스트
            [{'id': ..., 'text': ..., 'metadata': ..., 'distance': ...}, ...]
        """
        # SimpleVectorStore의 search 호출
        results = self.collection.search(
            query_embedding=query_embedding,
            k=k,
            filters=filters
        )
        
        return results
    
    def _build_where_filter(self, filters: Dict) -> Dict:
        """
        필터를 ChromaDB where 절로 변환
        
        Args:
            filters: {'category': '민사소송법', 'year_range': (2015, 2022)}
        
        Returns:
            ChromaDB where 절
        """
        where = {}
        
        for key, value in filters.items():
            if key == 'year_range' and isinstance(value, tuple):
                # 연도 범위 필터
                where['publication_year'] = {
                    '$gte': value[0],
                    '$lte': value[1]
                }
            elif isinstance(value, dict):
                # 이미 연산자가 포함된 경우
                where[key] = value
            else:
                # 단순 일치
                where[key] = value
        
        return where if where else None
    
    def delete_collection(self):
        """컬렉션 삭제"""
        self.collection.delete_collection()
        print(f"Deleted collection: {self.collection_name}")
    
    def get_count(self) -> int:
        """저장된 문서 수 반환"""
        return self.collection.get_count()


# 사용 예시
if __name__ == '__main__':
    import numpy as np
    
    # 테스트용 벡터 스토어
    vector_store = LegalVectorStore(
        persist_directory="./test_chroma_db",
        collection_name="test_legal"
    )
    
    # 더미 데이터 추가
    texts = ["대법원 판결 1", "대법원 판결 2"]
    embeddings = [np.random.rand(768).tolist() for _ in range(2)]
    metadatas = [
        {'book_id': 'TEST001', 'category': '민사소송법', 'publication_year': 2015},
        {'book_id': 'TEST002', 'category': '형법', 'publication_year': 2020}
    ]
    ids = ['doc1', 'doc2']
    
    vector_store.add_documents(texts, embeddings, metadatas, ids)
    
    # 검색 테스트
    query_embedding = np.random.rand(768).tolist()
    results = vector_store.search(query_embedding, k=2)
    
    print(f"\n검색 결과: {len(results)}개")
    for result in results:
        print(f"  - {result['metadata']['book_id']}: {result['text'][:30]}...")
    
    # 정리
    vector_store.delete_collection()

