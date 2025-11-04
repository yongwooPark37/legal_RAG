"""
데이터셋의 기본적인 통계와 특성을 분석 (EDA)
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import Dict, List, Tuple

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

def load_json_data(file_path: str) -> Dict:
    """
    JSON 파일 로드
    """
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_text_statistics(texts: List[str]) -> Dict:
    """
    텍스트 통계 분석
    """
    lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    
    return {
        'num_documents': len(texts),
        'char_length': {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'percentiles': {
                '25': np.percentile(lengths, 25),
                '50': np.percentile(lengths, 50),
                '75': np.percentile(lengths, 75),
                '90': np.percentile(lengths, 90),
                '95': np.percentile(lengths, 95),
                '99': np.percentile(lengths, 99)
            }
        },
        'word_count': {
            'mean': np.mean(word_counts),
            'median': np.median(word_counts),
            'std': np.std(word_counts),
            'min': np.min(word_counts),
            'max': np.max(word_counts),
        }
    }

def analyze_ner_statistics(data: List[Dict]) -> Dict:
    """
    NER 통계 분석
    """
    all_ner_types = []
    ner_counts_per_doc = []
    entity_lengths = defaultdict(list)
    
    for item in data:
        ne_list = item.get('NE', [])
        ner_counts_per_doc.append(len(ne_list))
        
        for ne in ne_list:
            ne_type = ne.get('type', 'UNKNOWN')
            all_ner_types.append(ne_type)
            entity_text = ne.get('entity', '')
            entity_lengths[ne_type].append(len(entity_text))
    
    ner_type_dist = Counter(all_ner_types)
    
    return {
        'total_entities': len(all_ner_types),
        'unique_entity_types': len(ner_type_dist),
        'entity_type_distribution': dict(ner_type_dist),
        'entities_per_document': {
            'mean': np.mean(ner_counts_per_doc),
            'median': np.median(ner_counts_per_doc),
            'std': np.std(ner_counts_per_doc),
            'min': np.min(ner_counts_per_doc),
            'max': np.max(ner_counts_per_doc),
        },
        'entity_length_by_type': {k: {
            'mean': np.mean(v),
            'median': np.median(v),
            'std': np.std(v)
        } for k, v in entity_lengths.items()}
    }

def analyze_category_statistics(data: List[Dict]) -> Dict:
    """
    카테고리 통계 분석
    """
    categories = [item.get('category', 'UNKNOWN') for item in data]
    category_dist = Counter(categories)
    
    return {
        'unique_categories': len(category_dist),
        'category_distribution': dict(category_dist),
        'category_percentages': {k: (v/len(categories))*100 for k, v in category_dist.items()}
    }

def analyze_keyword_statistics(data: List[Dict]) -> Dict:
    """
    키워드 통계 분석
    """
    all_keywords = []
    keyword_counts_per_doc = []
    
    for item in data:
        keywords = item.get('keyword', [])
        keyword_counts_per_doc.append(len(keywords))
        all_keywords.extend(keywords)
    
    keyword_dist = Counter(all_keywords)
    
    return {
        'total_keywords': len(all_keywords),
        'unique_keywords': len(keyword_dist),
        'top_keywords': dict(keyword_dist.most_common(50)),
        'keywords_per_document': {
            'mean': np.mean(keyword_counts_per_doc),
            'median': np.median(keyword_counts_per_doc),
            'std': np.std(keyword_counts_per_doc),
        }
    }

def analyze_publication_dates(data: List[Dict]) -> Dict:
    """
    출판일 통계 분석
    """
    dates = []
    years = []
    
    for item in data:
        pub_date = item.get('publication_ymd', '')
        if pub_date and len(pub_date) >= 4:
            dates.append(pub_date)
            years.append(int(pub_date[:4]))
    
    year_dist = Counter(years)
    
    return {
        'total_with_dates': len(dates),
        'year_distribution': dict(sorted(year_dist.items())),
        'year_range': {
            'min': min(years) if years else None,
            'max': max(years) if years else None
        }
    }

def visualize_text_length_distribution(text_lengths: List[int], save_path: str = 'eda_results/text_length_dist.png'):
    """
    텍스트 길이 분포 시각화
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 히스토그램
    axes[0].hist(text_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('텍스트 길이 (문자 수)')
    axes[0].set_ylabel('빈도')
    axes[0].set_title('텍스트 길이 분포')
    axes[0].grid(True, alpha=0.3)
    
    # 박스플롯
    axes[1].boxplot(text_lengths, vert=True)
    axes[1].set_ylabel('텍스트 길이 (문자 수)')
    axes[1].set_title('텍스트 길이 박스플롯')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def visualize_category_distribution(category_dist: Dict, save_path: str = 'eda_results/category_dist.png', top_n: int = 20):
    """
    카테고리 분포 시각화
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sorted_categories = sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:top_n]
    categories, counts = zip(*sorted_categories)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(categories)), counts, alpha=0.7)
    plt.yticks(range(len(categories)), categories)
    plt.xlabel('문서 수')
    plt.title(f'카테고리 분포 (상위 {top_n}개)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def visualize_ner_distribution(ner_dist: Dict, save_path: str = 'eda_results/ner_dist.png', top_n: int = 20):
    """
    NER 타입 분포 시각화
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    sorted_ner = sorted(ner_dist.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ner_types, counts = zip(*sorted_ner)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(ner_types)), counts, alpha=0.7)
    plt.yticks(range(len(ner_types)), ner_types)
    plt.xlabel('엔티티 수')
    plt.title(f'NER 타입 분포 (상위 {top_n}개)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main():
    print("=" * 80)
    print("법률 특화 RAG 기반 LLM 시스템 - EDA 시작")
    print("=" * 80)
    
    # 데이터 경로 설정
    train_json = 'data/Training/02.라벨링데이터/Training_legal.json'
    val_json = 'data/Validation/02.라벨링데이터/Validation_legal.json'
    
    # 결과 저장 디렉토리 생성
    os.makedirs('eda_results', exist_ok=True)
    
    # Training 데이터 로드 및 분석
    print("\n[Training 데이터 분석]")
    train_data_dict = load_json_data(train_json)
    train_data = train_data_dict.get('data', [])
    train_texts = [item.get('text', '') for item in train_data]
    
    print(f"Training 데이터 수: {train_data_dict.get('totalcount', len(train_data))}")
    
    # Validation 데이터 로드 및 분석
    print("\n[Validation 데이터 분석]")
    val_data_dict = load_json_data(val_json)
    val_data = val_data_dict.get('data', [])
    val_texts = [item.get('text', '') for item in val_data]
    
    print(f"Validation 데이터 수: {val_data_dict.get('totalcount', len(val_data))}")
    
    # 통계 분석
    print("\n[텍스트 통계 분석]")
    train_text_stats = analyze_text_statistics(train_texts)
    val_text_stats = analyze_text_statistics(val_texts)
    
    print("\nTraining 데이터 텍스트 통계:")
    print(f"  문서 수: {train_text_stats['num_documents']}")
    print(f"  평균 길이: {train_text_stats['char_length']['mean']:.2f} 문자")
    print(f"  중앙값 길이: {train_text_stats['char_length']['median']:.2f} 문자")
    print(f"  최소/최대: {train_text_stats['char_length']['min']} / {train_text_stats['char_length']['max']} 문자")
    
    print("\nValidation 데이터 텍스트 통계:")
    print(f"  문서 수: {val_text_stats['num_documents']}")
    print(f"  평균 길이: {val_text_stats['char_length']['mean']:.2f} 문자")
    print(f"  중앙값 길이: {val_text_stats['char_length']['median']:.2f} 문자")
    
    # NER 통계 분석
    print("\n[NER 통계 분석]")
    train_ner_stats = analyze_ner_statistics(train_data)
    val_ner_stats = analyze_ner_statistics(val_data)
    
    print("\nTraining 데이터 NER 통계:")
    print(f"  총 엔티티 수: {train_ner_stats['total_entities']}")
    print(f"  고유 엔티티 타입 수: {train_ner_stats['unique_entity_types']}")
    print(f"  문서당 평균 엔티티 수: {train_ner_stats['entities_per_document']['mean']:.2f}")
    print(f"\n  엔티티 타입 분포 (상위 10개):")
    top_ner_types = sorted(train_ner_stats['entity_type_distribution'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
    for ner_type, count in top_ner_types:
        print(f"    {ner_type}: {count}")
    
    # 카테고리 통계 분석
    print("\n[카테고리 통계 분석]")
    train_category_stats = analyze_category_statistics(train_data)
    
    print(f"\nTraining 데이터 카테고리 통계:")
    print(f"  고유 카테고리 수: {train_category_stats['unique_categories']}")
    print(f"  카테고리 분포 (상위 10개):")
    top_categories = sorted(train_category_stats['category_distribution'].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
    for category, count in top_categories:
        percentage = train_category_stats['category_percentages'][category]
        print(f"    {category}: {count} ({percentage:.2f}%)")
    
    # 키워드 통계 분석
    print("\n[키워드 통계 분석]")
    train_keyword_stats = analyze_keyword_statistics(train_data)
    
    print(f"\nTraining 데이터 키워드 통계:")
    print(f"  총 키워드 수: {train_keyword_stats['total_keywords']}")
    print(f"  고유 키워드 수: {train_keyword_stats['unique_keywords']}")
    print(f"  문서당 평균 키워드 수: {train_keyword_stats['keywords_per_document']['mean']:.2f}")
    print(f"\n  상위 키워드 (상위 20개):")
    top_keywords = list(train_keyword_stats['top_keywords'].items())[:20]
    for keyword, count in top_keywords:
        print(f"    {keyword}: {count}")
    
    # 출판일 통계 분석
    print("\n[출판일 통계 분석]")
    train_date_stats = analyze_publication_dates(train_data)
    
    print(f"\nTraining 데이터 출판일 통계:")
    print(f"  날짜 정보가 있는 문서 수: {train_date_stats['total_with_dates']}")
    if train_date_stats['year_range']['min']:
        print(f"  연도 범위: {train_date_stats['year_range']['min']} - {train_date_stats['year_range']['max']}")
        print(f"  연도별 분포 (최근 5년):")
        recent_years = sorted(train_date_stats['year_distribution'].items(), reverse=True)[:5]
        for year, count in recent_years:
            print(f"    {year}: {count}")
    
    # 시각화
    print("\n[시각화 생성]")
    train_lengths = [len(text) for text in train_texts]
    visualize_text_length_distribution(train_lengths, 'eda_results/train_text_length_dist.png')
    visualize_category_distribution(train_category_stats['category_distribution'], 
                                   'eda_results/train_category_dist.png')
    visualize_ner_distribution(train_ner_stats['entity_type_distribution'], 
                              'eda_results/train_ner_dist.png')
    
    # 결과를 JSON으로 저장
    results = {
        'train_text_stats': train_text_stats,
        'val_text_stats': val_text_stats,
        'train_ner_stats': {
            'total_entities': train_ner_stats['total_entities'],
            'unique_entity_types': train_ner_stats['unique_entity_types'],
            'entity_type_distribution': train_ner_stats['entity_type_distribution'],
            'entities_per_document': train_ner_stats['entities_per_document']
        },
        'train_category_stats': train_category_stats,
        'train_keyword_stats': {
            'total_keywords': train_keyword_stats['total_keywords'],
            'unique_keywords': train_keyword_stats['unique_keywords'],
            'top_keywords': train_keyword_stats['top_keywords'],
            'keywords_per_document': train_keyword_stats['keywords_per_document']
        },
        'train_date_stats': train_date_stats
    }
    
    # numpy 타입을 Python 기본 타입으로 변환
    results = convert_numpy_types(results)
    
    with open('eda_results/eda_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nEDA 결과가 'eda_results' 폴더에 저장되었습니다.")
    print("=" * 80)
    print("EDA 완료!")
    print("=" * 80)

if __name__ == '__main__':
    main()

