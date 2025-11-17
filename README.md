# 법률 특화 RAG 기반 LLM 시스템

## 프로젝트 개요

국내 법률에 특화된 AI 서비스를 통해 일반인들이 법률 정보에 쉽게 접근하고 이해할 수 있도록 하는 것이 목표입니다.

### Motivation

1. **법률 정보 접근성 향상**: 일반인들의 법률적 문제 해결 능력 향상
2. **법률 업무 효율성 제고**: 법률 관련 업무의 효율성 및 정확도 향상
3. **RAG 기반 접근의 필요성**: 지속적으로 업데이트되고 변화하는 법률 영역이기에 static time의 fine-tuning보다는 RAG 기반 접근이 적합

### Dataset

- **Training**: 63,704개 법률 판례 문서
- **Validation**: 7,963개 법률 판례 문서
- **평균 길이**: 약 2,980자
- **카테고리**: 23개 법률 카테고리
- **NER 라벨**: 10개 엔티티 타입, 277만개 엔티티

## 프로젝트 구조

```
legal_RAG/
├── data/                           # 데이터셋 (gitignore)
│   ├── Training/
│   └── Validation/
├── src/                            # 소스 코드
│   ├── data_processing/
│   │   └── chunking.py            # 청킹 전략 (베이스라인/의미 기반)
│   ├── embedding/
│   │   └── embedding_model.py     # 임베딩 모델 (한국어 특화)
│   ├── vector_store/
│   │   └── vector_db.py           # 벡터 DB (ChromaDB)
│   ├── retrieval/
│   │   └── retriever.py           # 검색기 (Dense Retrieval)
│   ├── generation/
│   │   └── generator.py           # LLM 생성 (GPT-3.5)
│   └── evaluation/
│       ├── retriever_eval.py      # 검색 평가 (Hit@K, MRR)
│       └── generator_eval.py      # 생성 평가 (LLM-as-a-Judge)
├── scripts/
│   ├── build_vector_db.py         # 벡터 DB 구축
│   ├── baseline_rag.py            # 베이스라인 RAG 시스템
│   └── demo.py                    # 데모 스크립트
├── eda.py                          # EDA 스크립트
├── eda_results/                    # EDA 결과
├── chroma_db/                      # 벡터 DB (생성됨, gitignore)
├── requirements.txt
├── README.md                       # 이 파일
├── QUICKSTART.md                   # 빠른 시작 가이드
└── IMPLEMENTATION_ROADMAP.md       # 상세 로드맵
```

## 빠른 시작

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt
```

### 2. 벡터 DB 구축

```bash
# 테스트용 (100개 문서, 약 5분)
python scripts/build_vector_db.py --limit 100

# 전체 데이터 (오래 걸림; 4시간 이상)
python scripts/build_vector_db.py
```

### 3. RAG 시스템 실행

```bash
# 데모 실행
python scripts/demo.py

# 대화형 모드
python scripts/baseline_rag.py

# 단일 질문
python scripts/baseline_rag.py --query "부동산 매매계약의 효력은?"
```


## 시스템 아키텍처

### 베이스라인 RAG 파이프라인

```
사용자 질문
    ↓
[1] Query Embedding (ko-sroberta-multitask)
    ↓
[2] Vector Search (ChromaDB, Top-K)
    ↓
[3] Retrieved Documents (5개 문서)
    ↓
[4] LLM Generation (GPT-3.5-turbo)
    ↓
답변 + 출처
```

### EDA 발견 → 설계 반영

| EDA 발견 | 설계 결정 | 근거 |
|----------|-----------|------|
| 평균 2,980자, 최대 133K자 | Chunking 1000자 | 긴 문서 처리 |
| 법률 용어 빈도 높음 | 한국어 특화 임베딩 | 도메인 특화 |
| 23개 카테고리 | Metadata 필터링 | 검색 정확도 향상 |
| 법률 구조 (판시사항 등) | Semantic Chunking | 맥락 보존 |

## 완료된 작업

1. **EDA**: 데이터 특성 파악 및 시각화
2. **데이터 전처리**: 베이스라인/의미 기반 chunking 구현
3. **임베딩**: 한국어 특화 임베딩 모델 적용
4. **벡터 DB**: ChromaDB 구축 (Metadata 지원)
5. **Retriever**: Dense Vector Search + 필터링
6. **Generator**: LLM 기반 답변 생성
7. **통합 시스템**: End-to-end RAG 파이프라인

## 추후 구현 사항

### 1: 평가 시스템 구축
- Validation 데이터로 검색 성능 평가 (Hit@K, MRR)
- LLM-as-a-Judge로 생성 품질 평가
- 베이스라인 성능 측정

### 2: 실험 및 개선
- 실험 1: Chunking 전략 비교
- 실험 2: Embedding 모델 비교
- 실험 3: Metadata 필터링 효과
- 실험 4: Hybrid Search + Reranking

### 3: Advanced things...
- 3-1. Context Engineering
- 3-2. AI Agent 구현: 복잡한 질문(예: "민법상 사기죄와 형법상 사기죄의 차이는?")은 RAG 한 번으로 답하기 어려움 -> LangChain의 Agent 사용

## 모델 관련 실험 방향

본 데이터셋은 KM-BERT NER, KL-BERT NER 기반으로 구성되었으나, 다양한 실험 가능:

1. **모델 다양성**: BERT, T5, Llama, Gemini, Qwen 등
2. **Augmentation Prompting**: 다양한 프롬프트 엔지니어링 기법
3. **RAG 구조 최적화**: 벡터 임베딩, 검색 전략, 재랭킹 등

