"""
임베딩 모델

EDA 근거: 법률 용어가 많으므로 한국어 특화 임베딩 모델 사용
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class LegalEmbeddingModel:
    """
    법률 문서 임베딩 모델
    
    베이스라인: 한국어 범용 임베딩 모델 사용
    """
    
    # 사용 가능한 모델들
    AVAILABLE_MODELS = {
        'baseline': 'jhgan/ko-sroberta-multitask',  # 베이스라인
        'simcse': 'BM-K/KoSimCSE-roberta-multitask',
        'multilingual': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'small': 'sentence-transformers/all-MiniLM-L6-v2'  # 빠른 테스트용
    }
    
    def __init__(self, model_name: str = 'baseline', device: str = None):
        """
        Args:
            model_name: 사용할 모델 ('baseline', 'simcse', 'multilingual', 'small')
            device: 'cuda', 'cpu', 또는 None (자동)
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_path = self.AVAILABLE_MODELS[model_name]
        
        print(f"Loading embedding model: {self.model_path}...")
        self.model = SentenceTransformer(self.model_path, device=device)
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress_bar: bool = True) -> np.ndarray:
        """
        텍스트를 벡터로 변환
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기
            show_progress_bar: 진행률 표시
        
        Returns:
            임베딩 벡터 (shape: [len(texts), embedding_dim])
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        단일 쿼리를 벡터로 변환
        
        Args:
            query: 검색 쿼리
        
        Returns:
            임베딩 벡터
        """
        return self.model.encode(query, convert_to_numpy=True)
    
    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.model.get_sentence_embedding_dimension()


# 사용 예시
if __name__ == '__main__':
    # 테스트
    embedder = LegalEmbeddingModel(model_name='baseline')
    
    texts = [
        "대법원 판결에 따르면 소유권이전등기는...",
        "민법 제110조는 다음과 같이 규정한다."
    ]
    
    embeddings = embedder.encode(texts, show_progress_bar=False)
    print(f"\n임베딩 shape: {embeddings.shape}")
    print(f"첫 번째 임베딩 (일부): {embeddings[0][:10]}")

