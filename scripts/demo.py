"""
테스트를 위한 데모 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.baseline_rag import BaselineRAG


def demo_queries():
    """데모 질문 리스트"""
    return [
        {
            'query': '부동산 매매계약의 합의해제 효력은 무엇인가요?',
            'category': None
        },
        {
            'query': '소유권이전등기와 관련된 판례를 알려주세요',
            'category': None
        },
        {
            'query': '형법상 사기죄의 구성요건은?',
            'category': '형법'
        },
        {
            'query': '부당해고에 대한 법적 구제방법은?',
            'category': '노동법'
        },
        {
            'query': '취득세 부과 요건은 무엇인가요?',
            'category': '조세/세법'
        }
    ]


def run_demo(vector_db_path: str = './chroma_db'):
    """
    데모 실행
    
    Args:
        vector_db_path: 벡터 DB 경로
    """
    print("=" * 80)
    print("법률 특화 RAG 시스템 데모")
    print("=" * 80)
    
    # RAG 시스템 초기화
    try:
        rag = BaselineRAG(
            vector_db_path=vector_db_path,
            embedding_model='baseline',
            llm_model='gpt-3.5-turbo'
        )
    except Exception as e:
        print(f"\nRAG 시스템 초기화 실패: {e}")
        print("\n먼저 벡터 DB를 구축해야 합니다:")
        print("  python scripts/build_vector_db.py --limit 100")
        return
    
    # 데모 질문 실행
    queries = demo_queries()
    
    print("\n" + "=" * 80)
    print("데모 질문 실행")
    print("=" * 80)
    
    for i, item in enumerate(queries, 1):
        print(f"\n\n{'=' * 80}")
        print(f"데모 {i}/{len(queries)}")
        print(f"{'=' * 80}")
        
        try:
            result = rag.answer(
                query=item['query'],
                category=item['category'],
                k=3,  # 데모용으로 3개만
                verbose=True
            )
            
            # 결과 요약
            print(f"\n✓ 성공: {result['contexts_used']}개 문서 사용")
            
        except Exception as e:
            print(f"\n✗ 오류: {e}")
        
        # 다음 질문으로 넘어가기
        if i < len(queries):
            input("\n계속하려면 Enter를 누르세요...")
    
    print("\n\n" + "=" * 80)
    print("데모 완료!")
    print("=" * 80)
    
    # 대화형 모드 제안
    print("\n대화형 모드를 시작하시겠습니까? (y/n): ", end='')
    choice = input().strip().lower()
    
    if choice == 'y':
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
                
                # 카테고리 입력 (선택)
                print("카테고리 필터 (선택, 없으면 Enter): ", end='')
                category = input().strip() or None
                
                result = rag.answer(query, k=5, category=category, verbose=True)
                
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {e}")
                continue


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG 시스템 데모')
    parser.add_argument('--db', type=str, default='./chroma_db',
                       help='벡터 DB 경로')
    
    args = parser.parse_args()
    
    run_demo(vector_db_path=args.db)

