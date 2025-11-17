"""
법률 문서 청킹 전략
EDA 발견: 평균 2,980자, 최대 133,165자 → 청킹 필수
"""

from typing import List, Dict
import re


class RecursiveCharacterTextSplitter:
    """텍스트 분할기"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=100, 
                 length_function=len, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """텍스트를 재귀적으로 분할"""
        chunks = []
        
        # 구분자로 분할 시도
        for separator in self.separators:
            if separator == "":
                # 마지막 구분자: 강제 분할
                return self._split_by_size(text)
            
            splits = text.split(separator)
            
            # 분할된 조각들을 청크 크기에 맞게 조합
            current_chunk = []
            current_size = 0
            
            for split in splits:
                split_size = len(split)
                
                if current_size + split_size > self.chunk_size and current_chunk:
                    # 현재 청크 완성
                    chunks.append(separator.join(current_chunk))
                    
                    # Overlap 처리
                    overlap_start = max(0, len(current_chunk) - 1)
                    current_chunk = current_chunk[overlap_start:]
                    current_size = sum(len(c) for c in current_chunk)
                
                current_chunk.append(split)
                current_size += split_size
            
            # 남은 청크 추가
            if current_chunk:
                chunks.append(separator.join(current_chunk))
            
            # 모든 청크가 크기 제한 내라면 완료
            if all(len(c) <= self.chunk_size * 1.5 for c in chunks):
                return [c for c in chunks if c.strip()]
        
        return chunks
    
    def _split_by_size(self, text: str) -> List[str]:
        """강제로 크기별 분할"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks


class LegalDocumentChunker:
    """
    법률 문서 청킹 전략
    
    EDA 결과를 반영한 설계:
    - 문서가 매우 길어서 청킹 필요
    - 법률 구조(판시사항, 판결요지 등) 고려
    """
    
    def __init__(self, strategy='baseline'):
        self.strategy = strategy
        self.section_patterns = {
            'judgment_summary': r'【판시사항】',
            'judgment_point': r'【판결요지】',
            'reference_law': r'【참조조문】',
            'original_decision': r'【원심판결】',
            'main_text': r'【주문】',
            'reason': r'【이유】'
        }
    
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        문서를 청크로 분할
        
        Args:
            text: 원본 텍스트
            metadata: 문서 메타데이터 (book_id, category 등)
        
        Returns:
            청크 리스트 [{'text': ..., 'metadata': {...}}]
        """
        if self.strategy == 'baseline':
            return self.baseline_chunking(text, metadata)
        elif self.strategy == 'semantic':
            return self.semantic_chunking(text, metadata)
        elif self.strategy == 'small':
            return self.baseline_chunking(text, metadata, chunk_size=500, overlap=50)
        elif self.strategy == 'large':
            return self.baseline_chunking(text, metadata, chunk_size=2000, overlap=200)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def baseline_chunking(self, text: str, metadata: Dict = None, 
                         chunk_size=1000, overlap=100) -> List[Dict]:
        """
        베이스라인: RecursiveCharacterTextSplitter
        
        EDA 근거: 평균 2,980자 → 1000자 청크가 적절
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        return [
            {
                'text': chunk,
                'metadata': {
                    **(metadata or {}),
                    'chunk_index': i,
                    'chunk_strategy': 'baseline',
                    'chunk_size': chunk_size
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def semantic_chunking(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        개선안: 법률 문서 구조 인식 청킹
        
        EDA 근거: 법률 문서는 명확한 구조가 있음
        - 판시사항, 판결요지, 참조조문, 이유 등
        """
        chunks = []
        sections = self._extract_sections(text)
        
        for section_type, section_text in sections:
            # 섹션이 너무 길면 추가 분할
            if len(section_text) > 2000:
                # 법 조항 단위로 분할 시도
                sub_chunks = self._split_by_provisions(section_text)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'text': sub_chunk,
                        'metadata': {
                            **(metadata or {}),
                            'chunk_index': len(chunks),
                            'chunk_strategy': 'semantic',
                            'section_type': section_type,
                            'sub_chunk': i
                        }
                    })
            else:
                chunks.append({
                    'text': section_text,
                    'metadata': {
                        **(metadata or {}),
                        'chunk_index': len(chunks),
                        'chunk_strategy': 'semantic',
                        'section_type': section_type
                    }
                })
        
        return chunks
    
    def _extract_sections(self, text: str) -> List[tuple]:
        """
        법률 문서의 섹션 추출
        
        Returns:
            [(section_type, section_text), ...]
        """
        sections = []
        section_markers = []
        
        # 섹션 마커 위치 찾기
        for section_type, pattern in self.section_patterns.items():
            for match in re.finditer(pattern, text):
                section_markers.append((match.start(), section_type))
        
        # 위치순 정렬
        section_markers.sort()
        
        # 섹션 텍스트 추출
        for i, (start, section_type) in enumerate(section_markers):
            end = section_markers[i + 1][0] if i + 1 < len(section_markers) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_type, section_text))
        
        # 섹션이 없으면 전체를 하나의 섹션으로
        if not sections:
            sections.append(('full_text', text))
        
        return sections
    
    def _split_by_provisions(self, text: str) -> List[str]:
        """
        법 조항 단위로 분할
        예: "제1조", "제1항", "1." 등
        """
        # 조항 패턴
        provision_pattern = r'(?:제\s*\d+\s*조|제\s*\d+\s*항|\d+\.)'
        
        # 조항 위치 찾기
        splits = []
        for match in re.finditer(provision_pattern, text):
            splits.append(match.start())
        
        if not splits:
            # 조항이 없으면 고정 크기로 분할
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150
            )
            return splitter.split_text(text)
        
        # 조항 기준으로 청크 생성
        chunks = []
        for i in range(len(splits)):
            start = splits[i]
            end = splits[i + 1] if i + 1 < len(splits) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        
        return chunks


# 사용 예시
if __name__ == '__main__':
    # 테스트용
    sample_text = """취득세등부과처분취소 (대법원 2015. 1. 15. 선고 2011두28714 판결) 
    【판시사항】 甲 주식회사가 乙 등과 부동산 매매계약을 체결하고...
    【판결요지】 甲 주식회사가 乙 등과...
    【참조조문】 구 지방세법..."""
    
    chunker = LegalDocumentChunker(strategy='semantic')
    chunks = chunker.chunk_document(sample_text, metadata={'book_id': 'TEST001'})
    
    print(f"생성된 청크 수: {len(chunks)}")
    for chunk in chunks:
        print(f"- {chunk['metadata']['section_type']}: {len(chunk['text'])} 문자")

