"""
벡터 데이터베이스 구축 스크립트

전체 법률 데이터를 전처리하고 벡터 DB에 저장
"""

import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.chunking import LegalDocumentChunker
from src.embedding.embedding_model import LegalEmbeddingModel
from src.vector_store.vector_db import LegalVectorStore


def load_legal_data(json_path: str):
    """법률 데이터 로드"""
    print(f"Loading data from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = data.get('data', [])
    print(f"Loaded {len(documents)} documents")
    
    return documents


def build_vector_database(
    data_path: str = 'data/Training/02.라벨링데이터/Training_legal.json',
    persist_dir: str = './chroma_db',
    collection_name: str = 'legal_documents',
    chunking_strategy: str = 'baseline',
    embedding_model: str = 'baseline',
    limit: int = None
):
    """
    벡터 데이터베이스 구축
    
    Args:
        data_path: 법률 데이터 JSON 경로
        persist_dir: 벡터 DB 저장 경로
        collection_name: 컬렉션 이름
        chunking_strategy: 청킹 전략 ('baseline', 'semantic', 'small', 'large')
        embedding_model: 임베딩 모델 ('baseline', 'simcse', 'multilingual', 'small')
        limit: 처리할 문서 수 제한 (None이면 전체)
    """
    
    print("=" * 80)
    print("법률 문서 벡터 DB 구축 시작")
    print("=" * 80)
    
    # 1. 데이터 로드
    documents = load_legal_data(data_path)
    
    if limit:
        documents = documents[:limit]
        print(f"제한: {limit}개 문서만 처리")
    
    # 2. 청킹
    print(f"\n[Step 1] 문서 청킹 (전략: {chunking_strategy})")
    chunker = LegalDocumentChunker(strategy=chunking_strategy)
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for doc in tqdm(documents, desc="Chunking"):
        # 문서 메타데이터
        metadata = {
            'book_id': doc.get('book_id', ''),
            'category': doc.get('category', ''),
            'publication_year': int(doc.get('publication_ymd', '0000')[:4]) if doc.get('publication_ymd') else 0,
            'keywords': ','.join(doc.get('keyword', [])[:5])  # 상위 5개 키워드
        }
        
        # 청킹
        chunks = chunker.chunk_document(doc.get('text', ''), metadata)
        
        for chunk in chunks:
            all_chunks.append(chunk['text'])
            all_metadatas.append(chunk['metadata'])
            
            # 고유 ID 생성
            book_id = chunk['metadata']['book_id']
            chunk_idx = chunk['metadata']['chunk_index']
            chunk_id = f"{book_id}_chunk{chunk_idx}"
            all_ids.append(chunk_id)
    
    print(f"총 {len(all_chunks)}개 청크 생성")
    
    # 3. 임베딩
    print(f"\n[Step 2] 임베딩 생성 (모델: {embedding_model})")
    embedder = LegalEmbeddingModel(model_name=embedding_model)
    
    # 배치로 임베딩 생성
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_embeddings = embedder.encode(batch_chunks, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings.tolist())
    
    print(f"임베딩 생성 완료: {len(all_embeddings)}개")
    
    # 4. 벡터 DB 저장
    print(f"\n[Step 3] 벡터 DB 저장 (경로: {persist_dir})")
    vector_store = LegalVectorStore(
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    
    # 기존 컬렉션이 있으면 삭제 (선택적)
    if vector_store.get_count() > 0:
        response = input(f"기존 컬렉션이 존재합니다 ({vector_store.get_count()}개 문서). 삭제하고 새로 만들까요? (y/n): ")
        if response.lower() == 'y':
            vector_store.delete_collection()
            vector_store = LegalVectorStore(persist_dir, collection_name)
        else:
            print("기존 컬렉션에 추가합니다.")
    
    vector_store.add_documents(
        texts=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    print(f"\n벡터 DB 구축 완료!")
    print(f"  - 총 문서 수: {len(documents)}")
    print(f"  - 총 청크 수: {len(all_chunks)}")
    print(f"  - 저장 경로: {persist_dir}")
    print(f"  - 컬렉션: {collection_name}")
    
    # 5. 테스트 검색
    print(f"\n[테스트] 샘플 검색")
    test_query = "부동산 매매계약"
    query_embedding = embedder.encode_query(test_query)
    results = vector_store.search(query_embedding.tolist(), k=3)
    
    print(f"쿼리: '{test_query}'")
    print(f"검색 결과: {len(results)}개\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['metadata']['category']}] {result['metadata']['book_id']}")
        print(f"   {result['text'][:100]}...")
        print(f"   Distance: {result['distance']:.4f}\n")
    
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='벡터 데이터베이스 구축')
    parser.add_argument('--data', type=str, 
                       default='data/Training/02.라벨링데이터/Training_legal.json',
                       help='법률 데이터 JSON 경로')
    parser.add_argument('--output', type=str, default='./chroma_db',
                       help='벡터 DB 저장 경로')
    parser.add_argument('--collection', type=str, default='legal_documents',
                       help='컬렉션 이름')
    parser.add_argument('--chunking', type=str, default='baseline',
                       choices=['baseline', 'semantic', 'small', 'large'],
                       help='청킹 전략')
    parser.add_argument('--embedding', type=str, default='baseline',
                       choices=['baseline', 'simcse', 'multilingual', 'small'],
                       help='임베딩 모델')
    parser.add_argument('--limit', type=int, default=None,
                       help='처리할 문서 수 제한 (테스트용)')
    
    args = parser.parse_args()
    
    build_vector_database(
        data_path=args.data,
        persist_dir=args.output,
        collection_name=args.collection,
        chunking_strategy=args.chunking,
        embedding_model=args.embedding,
        limit=args.limit
    )

