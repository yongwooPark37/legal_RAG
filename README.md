# 법률 특화 RAG 기반 LLM 시스템

대학교 딥러닝 수업 프로젝트: 법률 특화 RAG(Retrieval-Augmented Generation) 기반 LLM 시스템 고안

## 프로젝트 개요

국내 법률에 특화된 AI 서비스를 통해 일반인들이 법률 정보에 쉽게 접근하고 이해할 수 있도록 하는 것이 목표입니다.

### Motivation

1. **법률 정보 접근성 향상**: 일반인들의 법률적 문제 해결 능력 향상
2. **법률 업무 효율성 제고**: 법률 관련 업무의 효율성 및 정확도 향상
3. **RAG 기반 접근의 필요성**: 지속적으로 업데이트되고 변화하는 법률 영역이기에 static time의 fine-tuning보다는 RAG 기반 접근이 적합

### 데이터셋

- **Training**: `data/Training/02.라벨링데이터/Training_legal.json`
- **Validation**: `data/Validation/02.라벨링데이터/Validation_legal.json`
- **원천 데이터**: 법률 판례 텍스트 파일들 (LJU*.txt)
- **라벨링**: KM-BERT NER, KL-BERT NER 기반으로 라벨링된 데이터

## 프로젝트 구조

```
legal_RAG/
├── data/                    # 데이터셋
│   ├── Training/
│   └── Validation/
├── eda.py                   # Exploratory Data Analysis 스크립트
├── eda_results/             # EDA 결과 (생성됨)
├── requirements.txt         # Python 패키지 의존성
└── README.md               # 이 파일
```

## 시작하기

### 1. 환경 설정

```bash
# 가상 환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. EDA 실행

기본적인 데이터 분석을 수행합니다:

```bash
python eda.py
```

EDA 결과는 `eda_results/` 폴더에 저장됩니다:
- `eda_summary.json`: 통계 요약
- `*.png`: 시각화 결과

## 실험 방향

본 데이터셋은 KM-BERT NER, KL-BERT NER 기반으로 구성되었으나, 다양한 실험 가능:

1. **모델 다양성**: BERT, T5, Llama, Gemini, Qwen 등 다양한 모델 시도
2. **Augmentation Prompting**: 다양한 프롬프트 엔지니어링 기법 적용
3. **RAG 구조 최적화**: 벡터 임베딩, 검색 전략, 재랭킹 등

## 평가 기준

- 주제는 수업에서 다룬 딥러닝 주제 기반으로 자유롭게 선정
- 기존 연구 참고는 가능하나 반드시 팀만의 문제정의와 해석 필요
- 고민해야 할 점:
  - 이 문제가 왜 중요한지
  - 왜 이 방법을 선택했는지
  - 어떤 점에서 기존 연구 및 시도와 차별화되는지

## 다음 단계

1. ✅ EDA를 통한 데이터 특성 파악
2. ⏳ Baseline 모델 구축 및 평가
3. ⏳ 고유한 접근 방법 실험 및 개선
4. ⏳ 최종 결과 분석 및 보고서 작성

