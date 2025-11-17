"""
간단한 메모리 기반 벡터 스토어

ChromaDB 의존성 문제 해결을 위한 대안
"""

import numpy as np
import pickle
from typing import List, Dict, Optional
from pathlib import Path


class SimpleVectorStore:
    """
    메모리 기반 벡터 스토어
    
    Python 3.14 호환성 문제 해결을 위한 간단한 구현
    """
    
    def __init__(self, persist_directory: str = "./simple_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.texts = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
        # 기존 데이터 로드 시도
        self._load()
    
    def add_documents(self,
                     texts: List[str],
                     embeddings: List[List[float]],
                     metadatas: List[Dict],
                     ids: List[str]):
        """문서 추가"""
        
        for text, emb, meta, doc_id in zip(texts, embeddings, metadatas, ids):
            if doc_id in self.ids:
                # 이미 존재하면 업데이트
                idx = self.ids.index(doc_id)
                self.texts[idx] = text
                self.embeddings[idx] = np.array(emb)
                self.metadatas[idx] = meta
            else:
                # 새로 추가
                self.texts.append(text)
                self.embeddings.append(np.array(emb))
                self.metadatas.append(meta)
                self.ids.append(doc_id)
        
        # 자동 저장
        self._save()
        
        print(f"Added/Updated {len(texts)} documents (total: {len(self.ids)})")
    
    def search(self,
              query_embedding: List[float],
              k: int = 10,
              filters: Optional[Dict] = None) -> List[Dict]:
        """벡터 검색"""
        
        if not self.embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        
        # 코사인 유사도 계산
        embeddings_matrix = np.array(self.embeddings)
        
        # 정규화
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        embeddings_norm = embeddings_matrix / (np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-10)
        
        # 코사인 유사도
        similarities = np.dot(embeddings_norm, query_norm)
        
        # 필터링
        valid_indices = list(range(len(self.texts)))
        if filters:
            valid_indices = self._apply_filters(filters)
        
        # 유효한 인덱스의 유사도만 추출
        valid_similarities = [(idx, similarities[idx]) for idx in valid_indices]
        
        # 정렬
        sorted_indices = sorted(valid_similarities, key=lambda x: x[1], reverse=True)[:k]
        
        # 결과 포맷팅
        results = []
        for idx, similarity in sorted_indices:
            results.append({
                'id': self.ids[idx],
                'text': self.texts[idx],
                'metadata': self.metadatas[idx],
                'distance': 1 - similarity  # distance로 변환
            })
        
        return results
    
    def _apply_filters(self, filters: Dict) -> List[int]:
        """필터 적용"""
        valid_indices = []
        
        for idx, meta in enumerate(self.metadatas):
            match = True
            
            for key, value in filters.items():
                if key == 'year_range' and isinstance(value, tuple):
                    year = meta.get('publication_year', 0)
                    if year < value[0] or year > value[1]:
                        match = False
                        break
                elif isinstance(value, dict):
                    # 연산자 지원 ($gte, $lte 등)
                    meta_value = meta.get(key, None)
                    if meta_value is None:
                        match = False
                        break
                    if '$gte' in value and meta_value < value['$gte']:
                        match = False
                        break
                    if '$lte' in value and meta_value > value['$lte']:
                        match = False
                        break
                else:
                    # 단순 일치
                    if meta.get(key) != value:
                        match = False
                        break
            
            if match:
                valid_indices.append(idx)
        
        return valid_indices
    
    def get_count(self) -> int:
        """문서 수 반환"""
        return len(self.ids)
    
    def delete_collection(self):
        """컬렉션 삭제"""
        self.texts = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
        # 파일 삭제
        db_file = self.persist_directory / "vector_store.pkl"
        if db_file.exists():
            db_file.unlink()
        
        print("Collection deleted")
    
    def _save(self):
        """디스크에 저장"""
        db_file = self.persist_directory / "vector_store.pkl"
        
        data = {
            'texts': self.texts,
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'metadatas': self.metadatas,
            'ids': self.ids
        }
        
        with open(db_file, 'wb') as f:
            pickle.dump(data, f)
    
    def _load(self):
        """디스크에서 로드"""
        db_file = self.persist_directory / "vector_store.pkl"
        
        if db_file.exists():
            with open(db_file, 'rb') as f:
                data = pickle.load(f)
            
            self.texts = data['texts']
            self.embeddings = [np.array(emb) for emb in data['embeddings']]
            self.metadatas = data['metadatas']
            self.ids = data['ids']
            
            print(f"Loaded existing collection: {len(self.ids)} documents")


# 사용 예시
if __name__ == '__main__':
    # 테스트
    store = SimpleVectorStore("./test_simple_db")
    
    texts = ["대법원 판결 1", "대법원 판결 2"]
    embeddings = [np.random.rand(768).tolist() for _ in range(2)]
    metadatas = [
        {'book_id': 'TEST001', 'category': '민사소송법'},
        {'book_id': 'TEST002', 'category': '형법'}
    ]
    ids = ['doc1', 'doc2']
    
    store.add_documents(texts, embeddings, metadatas, ids)
    
    # 검색
    query_emb = np.random.rand(768).tolist()
    results = store.search(query_emb, k=2)
    
    print(f"\n검색 결과: {len(results)}개")
    for r in results:
        print(f"  - {r['id']}: {r['text']}")

