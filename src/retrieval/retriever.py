"""
검색기 (Retriever)
베이스라인: Dense Retrieval (벡터 검색)만 사용
"""

from typing import List, Dict, Optional


class LegalRetriever:
    """
    법률 문서 검색기
    
    베이스라인: Vector Search Only
    향후 개선: Hybrid (BM25 + Vector), Reranking
    """
    
    def __init__(self, vector_store, embedding_model):
        """
        Args:
            vector_store: LegalVectorStore 인스턴스
            embedding_model: LegalEmbeddingModel 인스턴스
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def retrieve(self, 
                query: str, 
                k: int = 10,
                category: Optional[str] = None,
                year_range: Optional[tuple] = None) -> List[Dict]:
        """
        쿼리에 대한 관련 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            category: 카테고리 필터 (예: '민사소송법')
            year_range: 연도 범위 필터 (예: (2015, 2022))
        
        Returns:
            검색 결과 리스트
            [{'id': ..., 'text': ..., 'metadata': ..., 'score': ...}, ...]
        """
        # 1. 쿼리 임베딩
        query_embedding = self.embedding_model.encode_query(query)
        
        # 2. 필터 구성
        filters = {}
        if category:
            filters['category'] = category
        if year_range:
            filters['year_range'] = year_range
        
        # 3. 벡터 검색
        results = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            k=k,
            filters=filters if filters else None
        )
        
        # 4. 결과 포맷팅 (distance를 score로 변환)
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result['id'],
                'text': result['text'],
                'metadata': result['metadata'],
                'score': 1 - result['distance'],  # distance를 similarity로 변환
                'distance': result['distance']
            })
        
        return formatted_results
    
    def retrieve_with_context(self, 
                             query: str, 
                             k: int = 5,
                             category: Optional[str] = None) -> Dict:
        """
        검색 결과와 함께 컨텍스트 정보 반환
        
        LLM에 전달하기 좋은 형태로 포맷팅
        
        Returns:
            {
                'query': 원본 쿼리,
                'contexts': [텍스트1, 텍스트2, ...],
                'sources': [메타데이터1, 메타데이터2, ...],
                'results': 전체 검색 결과
            }
        """
        results = self.retrieve(query, k=k, category=category)
        
        contexts = [r['text'] for r in results]
        sources = [r['metadata'] for r in results]
        
        return {
            'query': query,
            'contexts': contexts,
            'sources': sources,
            'results': results
        }


# 사용 예시
if __name__ == '__main__':
    from src.embedding.embedding_model import LegalEmbeddingModel
    from src.vector_store.vector_db import LegalVectorStore
    import numpy as np
    
    # 임베딩 모델 로드
    embedder = LegalEmbeddingModel(model_name='small')  # 빠른 테스트용
    
    # 벡터 스토어 생성
    vector_store = LegalVectorStore(
        persist_directory="./test_chroma_db",
        collection_name="test_retriever"
    )
    
    # 더미 데이터 추가
    texts = [
        "대법원은 소유권이전등기에 관한 판결을 내렸다.",
        "형법 제110조는 사기죄에 대해 규정한다.",
        "노동법에 따르면 해고는 정당한 사유가 있어야 한다."
    ]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    metadatas = [
        {'book_id': 'TEST001', 'category': '민사소송법'},
        {'book_id': 'TEST002', 'category': '형법'},
        {'book_id': 'TEST003', 'category': '노동법'}
    ]
    ids = ['doc1', 'doc2', 'doc3']
    
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    
    # Retriever 생성 및 테스트
    retriever = LegalRetriever(vector_store, embedder)
    
    # 검색 테스트
    query = "소유권 이전에 관한 판례"
    results = retriever.retrieve(query, k=2)
    
    print(f"\n쿼리: {query}")
    print(f"검색 결과: {len(results)}개\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['metadata']['category']}] (score: {result['score']:.3f})")
        print(f"   {result['text'][:50]}...\n")
    
    # 카테고리 필터링 테스트
    print("\n카테고리 필터링 테스트 (형법만):")
    filtered_results = retriever.retrieve(query, k=5, category='형법')
    for result in filtered_results:
        print(f"  - {result['metadata']['category']}: {result['text'][:30]}...")
    
    # 정리
    vector_store.delete_collection()

