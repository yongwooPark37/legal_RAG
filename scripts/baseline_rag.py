"""
베이스라인 RAG 시스템

전체 RAG 파이프라인을 하나로 통합
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embedding.embedding_model import LegalEmbeddingModel
from src.vector_store.vector_db import LegalVectorStore
from src.retrieval.retriever import LegalRetriever
from src.generation.generator import LegalGenerator


class BaselineRAG:
    """
    베이스라인 RAG 시스템
    
    구성:
    - Retriever: Dense Vector Search
    - Generator: GPT-3.5-turbo
    """
    
    def __init__(self, 
                 vector_db_path: str = './chroma_db',
                 collection_name: str = 'legal_documents',
                 embedding_model: str = 'baseline',
                 llm_model: str = 'gpt-3.5-turbo'):
        """
        Args:
            vector_db_path: 벡터 DB 경로
            collection_name: 컬렉션 이름
            embedding_model: 임베딩 모델
            llm_model: LLM 모델
        """
        print("=" * 80)
        print("베이스라인 RAG 시스템 초기화")
        print("=" * 80)
        
        # 1. 임베딩 모델 로드
        print(f"\n[1] 임베딩 모델 로드: {embedding_model}")
        self.embedding_model = LegalEmbeddingModel(model_name=embedding_model)
        
        # 2. 벡터 스토어 로드
        print(f"\n[2] 벡터 DB 로드: {vector_db_path}")
        self.vector_store = LegalVectorStore(
            persist_directory=vector_db_path,
            collection_name=collection_name
        )
        
        # 3. Retriever 초기화
        print(f"\n[3] Retriever 초기화")
        self.retriever = LegalRetriever(self.vector_store, self.embedding_model)
        
        # 4. Generator 초기화
        print(f"\n[4] Generator 초기화: {llm_model}")
        self.generator = LegalGenerator(model=llm_model)
        
        print("\n초기화 완료!")
        print("=" * 80)
    
    def answer(self, 
              query: str, 
              k: int = 5,
              category: str = None,
              verbose: bool = True) -> dict:
        """
        질문에 답변
        
        Args:
            query: 사용자 질문
            k: 검색할 문서 수
            category: 카테고리 필터 (선택)
            verbose: 상세 출력 여부
        
        Returns:
            {
                'query': 질문,
                'answer': 답변,
                'sources': 출처,
                'retrieval_results': 검색 결과
            }
        """
        if verbose:
            print(f"\n질문: {query}")
            if category:
                print(f"카테고리 필터: {category}")
            print("-" * 80)
        
        # 1. 문서 검색
        if verbose:
            print(f"\n[검색 중...] 상위 {k}개 문서 검색")
        
        retrieval_result = self.retriever.retrieve_with_context(
            query=query,
            k=k,
            category=category
        )
        
        contexts = retrieval_result['contexts']
        sources = retrieval_result['sources']
        
        if verbose:
            print(f"검색 완료: {len(contexts)}개 문서")
            for i, src in enumerate(sources, 1):
                print(f"  {i}. [{src.get('category', 'N/A')}] {src.get('book_id', 'N/A')}")
        
        # 2. 답변 생성
        if verbose:
            print(f"\n[답변 생성 중...]")
        
        generation_result = self.generator.generate(
            query=query,
            contexts=contexts,
            sources=sources,
            return_sources=True
        )
        
        answer = generation_result['answer']
        
        if verbose:
            print(f"\n답변:")
            print("-" * 80)
            print(answer)
            print("-" * 80)
            
            if 'sources' in generation_result:
                print(f"\n참조 출처:")
                for src in generation_result['sources']:
                    print(f"  - {src}")
        
        # 결과 반환
        return {
            'query': query,
            'answer': answer,
            'sources': generation_result.get('sources', []),
            'retrieval_results': retrieval_result['results'],
            'contexts_used': generation_result.get('contexts_used', 0)
        }


# 사용 예시
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='베이스라인 RAG 시스템')
    parser.add_argument('--db', type=str, default='./chroma_db',
                       help='벡터 DB 경로')
    parser.add_argument('--collection', type=str, default='legal_documents',
                       help='컬렉션 이름')
    parser.add_argument('--embedding', type=str, default='baseline',
                       help='임베딩 모델')
    parser.add_argument('--llm', type=str, default='gpt-3.5-turbo',
                       help='LLM 모델')
    parser.add_argument('--query', type=str, default=None,
                       help='질문 (없으면 대화형 모드)')
    parser.add_argument('--category', type=str, default=None,
                       help='카테고리 필터')
    parser.add_argument('--k', type=int, default=5,
                       help='검색할 문서 수')
    
    args = parser.parse_args()
    
    # RAG 시스템 초기화
    rag = BaselineRAG(
        vector_db_path=args.db,
        collection_name=args.collection,
        embedding_model=args.embedding,
        llm_model=args.llm
    )
    
    # 질문이 주어지면 한 번만 실행
    if args.query:
        result = rag.answer(args.query, k=args.k, category=args.category)
    else:
        # 대화형 모드
        print("\n" + "=" * 80)
        print("대화형 모드 (종료: 'exit' 또는 'quit')")
        print("=" * 80)
        
        while True:
            try:
                query = input("\n질문을 입력하세요: ").strip()
                
                if query.lower() in ['exit', 'quit', '종료']:
                    print("종료합니다.")
                    break
                
                if not query:
                    continue
                
                result = rag.answer(query, k=args.k, category=args.category)
                
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {e}")
                continue

