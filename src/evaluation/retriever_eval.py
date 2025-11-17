"""
Retriever 평가 시스템

Validation 데이터를 활용한 검색 성능 평가
"""

from typing import List, Dict
import numpy as np
from tqdm import tqdm


class RetrieverEvaluator:
    """
    검색 성능 평가
    
    EDA 근거: Validation 데이터 7,963개 활용
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def evaluate_on_validation(self, val_data: List[Dict], k_values=[1, 5, 10]) -> Dict:
        """
        Validation 데이터로 검색 성능 평가
        
        핵심 아이디어:
        1. 각 validation 문서의 키워드로 쿼리 생성
        2. 검색 결과에 원본 문서가 있는지 확인
        3. Hit@K, MRR 계산
        
        Args:
            val_data: Validation 데이터 리스트
            k_values: 평가할 K 값들
        
        Returns:
            {'hit@1': 0.65, 'hit@5': 0.82, 'hit@10': 0.91, 'mrr': 0.73}
        """
        hit_at_k = {k: 0 for k in k_values}
        mrr_scores = []
        category_correct = 0
        
        print(f"\n[Retriever 평가] {len(val_data)}개 문서 평가 중...")
        
        for doc in tqdm(val_data):
            # 1. 쿼리 생성 (키워드 기반)
            query = self._create_query_from_doc(doc)
            
            # 2. 검색 수행
            max_k = max(k_values)
            results = self.retriever.retrieve(query, k=max_k)
            
            # 3. Hit@K 및 MRR 계산
            doc_id = doc['book_id']
            doc_category = doc.get('category', '')
            
            for rank, result in enumerate(results, 1):
                result_id = result.get('book_id') or result.get('metadata', {}).get('book_id')
                
                if result_id == doc_id:
                    # 정확히 같은 문서를 찾음
                    mrr_scores.append(1.0 / rank)
                    
                    for k in k_values:
                        if rank <= k:
                            hit_at_k[k] += 1
                    
                    # 카테고리 일치 여부
                    result_category = result.get('category') or result.get('metadata', {}).get('category')
                    if result_category == doc_category:
                        category_correct += 1
                    
                    break
            else:
                # 찾지 못함
                mrr_scores.append(0.0)
        
        # 결과 계산
        n = len(val_data)
        results = {
            f'hit@{k}': hit_at_k[k] / n for k in k_values
        }
        results['mrr'] = np.mean(mrr_scores)
        results['category_accuracy'] = category_correct / n
        
        return results
    
    def _create_query_from_doc(self, doc: Dict) -> str:
        """
        문서에서 쿼리 생성
        
        전략:
        1. 키워드 활용 (상위 3개)
        2. 주요 엔티티 활용
        3. 카테고리 정보 활용 (선택적)
        """
        keywords = doc.get('keyword', [])
        
        # 상위 3개 키워드 조합
        query_parts = keywords[:3]
        
        # 카테고리 추가 (선택적)
        # category = doc.get('category', '')
        # if category:
        #     query_parts.append(category)
        
        query = ' '.join(query_parts)
        
        # 법률 문맥 추가
        if query:
            query += " 판례"
        
        return query
    
    def evaluate_with_synthetic_qa(self, qa_pairs: List[Dict]) -> Dict:
        """
        합성 Q&A로 평가 (LLM 생성)
        
        Args:
            qa_pairs: [{'query': ..., 'answer': ..., 'doc_id': ...}, ...]
        """
        results = []
        
        for qa in tqdm(qa_pairs, desc="Q&A 평가"):
            query = qa['query']
            expected_doc_id = qa['doc_id']
            
            # 검색
            retrieved = self.retriever.retrieve(query, k=10)
            
            # 정답 문서가 포함되었는지
            found = any(
                r.get('book_id') == expected_doc_id or 
                r.get('metadata', {}).get('book_id') == expected_doc_id
                for r in retrieved
            )
            
            results.append(1 if found else 0)
        
        return {
            'accuracy': np.mean(results),
            'total': len(qa_pairs)
        }


# 사용 예시
if __name__ == '__main__':
    # 테스트용 더미 데이터
    val_data = [
        {
            'book_id': 'LJU000001',
            'category': '민사소송법',
            'keyword': ['대법원', '판결', '소유권']
        }
    ]
    
    # 더미 retriever
    class DummyRetriever:
        def retrieve(self, query, k=10):
            return [{'book_id': 'LJU000001', 'category': '민사소송법'}]
    
    evaluator = RetrieverEvaluator(DummyRetriever())
    metrics = evaluator.evaluate_on_validation(val_data)
    
    print("\n평가 결과:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

