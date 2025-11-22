"""
Legal RAG Project - Final Ultimate EDA Script

[설정 요약]
- Log Scale 적용: 1번(문서 길이), 3번(토큰 수) -> 극단값 보정
- Linear Scale (원본): 나머지 6개 -> 데이터 그대로 표현
- 개선사항: 막대 그래프 위에 정확한 수치(Label) 표기
- 경고 해결: Palette/Hue Warning, Glyph Warning 해결

Target: Full Dataset
Theme: Earth Tones (Extended 20 Colors)
"""

import json
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from wordcloud import WordCloud
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# 🎨 Design System: Earth & Beige (20색 확장판)
# -----------------------------------------------------------------------------
COLORS = {
    'bg': '#F9F8F2',
    'text': '#4A4238',
    'sub_text': '#8C857B',
    'accent_safe': '#8FBC8F',
    'accent_risk': '#E2725B',
    'palette': [
        '#8FBC8F', '#D2B48C', '#CD853F', '#778899', '#BC8F8F', '#E2725B',
        '#A9A9A9', '#556B2F', '#8B4513', '#DAA520', '#5F9EA0', '#A0522D',
        '#6B8E23', '#BDB76B', '#4682B4', '#DEB887', '#2F4F4F', '#CD5C5C',
        '#808000', '#708090'
    ]
}

FONT_PATH = None

def set_plot_style():
    plt.rcParams['figure.facecolor'] = COLORS['bg']
    plt.rcParams['axes.facecolor'] = COLORS['bg']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 폰트 자동 설정
    import platform
    global FONT_PATH
    system_name = platform.system()
    if system_name == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
        FONT_PATH = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    elif system_name == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
        FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'
    
    sns.set_palette(sns.color_palette(COLORS['palette']))

# -----------------------------------------------------------------------------
# 🧠 Final Analysis Class
# -----------------------------------------------------------------------------

class FinalLegalEDA:
    def __init__(self, data_path, result_dir='eda_results'):
        self.data_path = data_path
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.docs = self._load_data()

    def _load_data(self):
        print(f"📂 데이터 로드 중: {self.data_path}")
        if not os.path.exists(self.data_path):
            print(f"❌ 오류: 파일을 찾을 수 없습니다.")
            sys.exit(1)
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('data', data) if isinstance(data, dict) else data
        print(f"✅ 총 {len(documents):,}건의 문서 로드 완료.\n")
        return documents

    # =========================================================================
    # Part 1. Basic Analysis (기초)
    # =========================================================================

    def analyze_01_length(self):
        """[1] 문서 길이 분석 (Log Scale)"""
        print("[1/8] 문서 길이 분석 중 (Log Scale 적용)...")
        lengths = [len(doc.get('text', '')) for doc in self.docs]
        
        plt.figure(figsize=(10, 6))
        # log_scale=True 적용
        sns.histplot(lengths, bins=50, color=COLORS['palette'][1], kde=True, log_scale=True)
        plt.axvline(np.mean(lengths), color=COLORS['text'], linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
        plt.title('문서 길이 분포 (Log Scale)', fontsize=15)
        plt.xlabel('글자 수 (Log Scale)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.result_dir / '01_basic_length.png', dpi=300)

    def analyze_02_structure(self):
        """[2] 문서 구조 분석 (Linear Scale + 수치 표시)"""
        print("[2/8] 문서 구조 패턴 분석 중...")
        pattern = r'【(.*?)】'
        all_headers = []
        
        # 노이즈 필터링 (너무 긴 헤더 제외)
        for doc in tqdm(self.docs, desc="Scanning"):
            headers = re.findall(pattern, doc.get('text', ''))
            all_headers.extend([h.strip() for h in headers if len(h) < 10])
            
        counts = Counter(all_headers).most_common(15)
        df = pd.DataFrame(counts, columns=['Section', 'Count'])
        
        plt.figure(figsize=(12, 6))
        # Warning 해결: hue 지정, legend=False
        ax = sns.barplot(data=df, x='Section', y='Count', hue='Section', palette=COLORS['palette'], legend=False)
        
        # 막대 위에 숫자 표시 (진짜 똑같은지 확인용)
        for i in ax.containers:
            ax.bar_label(i, fmt='%d', padding=3)
            
        plt.title('주요 문서 섹션 헤더 (Top 15)', fontsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.result_dir / '02_basic_structure.png', dpi=300)

    def analyze_03_tokens(self):
        """[3] 토큰 수 분석 (Log Scale)"""
        print("[3/8] 토큰 수 분석 중 (Log Scale 적용)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
            use_tokenizer = True
        except:
            use_tokenizer = False
            
        token_counts = []
        for doc in tqdm(self.docs, desc="Counting"):
            text = doc.get('text', '')
            if use_tokenizer:
                token_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
            else:
                token_counts.append(len(text.split()))
                
        plt.figure(figsize=(10, 6))
        # log_scale=True 적용
        sns.histplot(token_counts, bins=50, color=COLORS['palette'][3], kde=True, log_scale=True)
        plt.axvline(512, color=COLORS['accent_risk'], linestyle='--', label='512 Tokens')
        plt.title('토큰 수 분포 (Log Scale)', fontsize=15)
        plt.xlabel('토큰 수 (Log Scale)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.result_dir / '03_basic_tokens.png', dpi=300)

    def analyze_04_wordcloud(self):
        """[4] 워드 클라우드"""
        print("[4/8] 워드 클라우드 생성 중...")
        keywords = []
        for doc in self.docs:
            keywords.extend(doc.get('keyword', []))
        if not keywords: return

        count = Counter(keywords)
        wc = WordCloud(
            font_path=FONT_PATH, width=1200, height=800,
            background_color=COLORS['bg'], colormap='copper',
            max_words=100
        ).generate_from_frequencies(count)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('주요 법률 키워드', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.result_dir / '04_basic_wordcloud.png', dpi=300)

    # =========================================================================
    # Part 2. Advanced Analysis (심화)
    # =========================================================================

    def analyze_05_ne_distribution(self):
        """[5] 개체명(NE) 분포"""
        print("[5/8] 개체명(NE) 분포 분석 중...")
        types = []
        for doc in self.docs:
            if 'NE' in doc:
                types.extend([ne['type'] for ne in doc['NE']])
        if not types: return
        
        df = pd.DataFrame(Counter(types).items(), columns=['Type', 'Count']).sort_values('Count', ascending=False)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x='Count', y='Type', hue='Type', palette=COLORS['palette'], legend=False)
        
        # 수치 표시
        for i in ax.containers:
            ax.bar_label(i, fmt='%d', padding=3)
            
        plt.title('개체명(NE) 타입별 분포', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.result_dir / '05_adv_ne_dist.png', dpi=300)

    def analyze_06_risk(self):
        """[6] 청킹 위험도"""
        print("[6/8] 청킹 위험도 분석 중...")
        risk_ratios = []
        for doc in tqdm(self.docs, desc="Risk Calc"):
            text_len = len(doc.get('text', ''))
            if text_len == 0: continue
            ne_len = sum([ne['end'] - ne['begin'] for ne in doc.get('NE', [])])
            risk_ratios.append((ne_len / text_len) * 100)
            
        avg_risk = np.mean(risk_ratios)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(risk_ratios, bins=50, color=COLORS['accent_risk'], kde=True)
        plt.axvline(avg_risk, color=COLORS['text'], linestyle='--', label=f'Mean Risk: {avg_risk:.1f}%')
        plt.title('문서별 청킹 위험도 분포 (NE Ratio)', fontsize=15)
        plt.xlabel('위험도 (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.result_dir / '06_adv_risk.png', dpi=300)

    def analyze_07_heatmap(self):
        """[7] 엔티티 히트맵 (수정됨: Float -> Int 변환 추가)"""
        print("[7/8] 법령-판결 관계 분석 중...")
        pair_counts = defaultdict(int)
        stop_laws = ['헌법', '민법', '형법'] # 너무 흔한 법률 제외
        
        for doc in tqdm(self.docs, desc="Mapping"):
            laws = set([ne['entity'] for ne in doc.get('NE', []) if ne['type'] == 'CV_LAW'])
            judgments = set([ne['entity'] for ne in doc.get('NE', []) if ne['type'] == 'TML_JUDGMENT'])
            
            for l in laws:
                if l in stop_laws or len(l) < 2: continue
                for j in judgments:
                    if len(j) < 2: continue
                    short_l = l[:10] + '..' if len(l) > 10 else l
                    short_j = j[:6] + '..' if len(j) > 6 else j
                    pair_counts[(short_l, short_j)] += 1
                    
        if not pair_counts: return
        
        top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Pivot Table로 변환
        data = [{'Law': k[0], 'Judgment': k[1], 'Count': v} for k, v in top_pairs]
        df = pd.DataFrame(data)
        
        # 1. 피벗 테이블 생성
        matrix = df.pivot_table(index='Law', columns='Judgment', values='Count', fill_value=0)
        
        # 2. [핵심 수정] 실수(float)를 정수(int)로 강제 변환
        matrix = matrix.astype(int)
        
        plt.figure(figsize=(10, 8))
        # 이제 데이터가 int이므로 fmt='d'가 정상 작동합니다
        sns.heatmap(matrix, annot=True, fmt='d', cmap='OrRd')
        plt.title('주요 법령-판결 연관성 (Filtered)', fontsize=15)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.result_dir / '07_adv_heatmap.png', dpi=300)

    def analyze_08_grounding(self):
        """[8] 그라운딩"""
        print("[8/8] 키워드 그라운딩 검증 중...")
        match, miss = 0, 0
        for doc in tqdm(self.docs, desc="Grounding"):
            text = doc.get('text', '')
            for kw in doc.get('keyword', []):
                if kw in text: match += 1
                else: miss += 1
        total = match + miss
        if total == 0: return
        
        plt.figure(figsize=(6, 6))
        plt.pie([match, miss], labels=['Matched', 'Missed'], 
                colors=[COLORS['accent_safe'], COLORS['accent_risk']],
                autopct='%1.1f%%', startangle=90)
        plt.title(f'키워드-본문 일치율', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.result_dir / '08_adv_grounding.png', dpi=300)

    def run_all(self):
        print("="*60)
        print("🚀 Final Ultimate EDA Starting...")
        print("="*60)
        
        self.analyze_01_length()
        self.analyze_02_structure()
        self.analyze_03_tokens()
        self.analyze_04_wordcloud()
        self.analyze_05_ne_distribution()
        self.analyze_06_risk()
        self.analyze_07_heatmap()
        self.analyze_08_grounding()
        
        print("\n✨ 모든 분석 완료! 'eda_results' 폴더를 확인하세요.")

if __name__ == "__main__":
    set_plot_style()
    # 실제 데이터 경로 설정
    DATA_PATH = 'data/Training/02.라벨링데이터/Training_legal.json'
    eda = FinalLegalEDA(DATA_PATH)
    eda.run_all()