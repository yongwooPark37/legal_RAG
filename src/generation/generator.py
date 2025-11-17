"""
답변 생성기 (Generator)
LLM을 사용하여 검색된 문서를 바탕으로 답변 생성
"""

from typing import List, Dict, Optional
from openai import OpenAI
import os

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일의 환경변수 로드
except ImportError:
    pass  # python-dotenv가 없어도 환경변수는 사용 가능


class LegalGenerator:
    """
    법률 질문에 대한 답변 생성
    
    베이스라인: OpenAI GPT 사용
    프롬프트 엔지니어링으로 법률 전문가처럼 답변하도록 유도
    """
    
    PROMPT_TEMPLATE = """당신은 대한민국 법률 전문가입니다.

아래 <문서>를 **반드시 근거로 하여** 답변하세요.
- 문서에 근거가 없으면 "제공된 자료에서 관련 정보를 찾을 수 없습니다"라고 답하세요.
- 답변은 명확하고 이해하기 쉽게 작성하세요.
- 답변 마지막에 **참조 출처**를 명시하세요.

<문서>
{contexts}
</문서>

<질문>
{query}
</질문>

<답변>"""
    
    def __init__(self, model: str = 'gpt-3.5-turbo', api_key: Optional[str] = None):
        """
        Args:
            model: 사용할 LLM 모델
            api_key: OpenAI API 키 (None이면 환경 변수에서 읽음)
        """
        self.model = model
        
        # API 키 설정
        api_key_to_use = api_key or os.getenv('OPENAI_API_KEY', '')
        self.has_api_key = bool(api_key_to_use)
        
        if self.has_api_key:
            self.client = OpenAI(api_key=api_key_to_use)
        else:
            self.client = None
            print("Warning: OpenAI API 키가 설정되지 않았습니다.")
            print("환경 변수 OPENAI_API_KEY를 설정하거나 api_key 인자를 전달하세요.")
    
    def generate(self, 
                query: str, 
                contexts: List[str],
                sources: Optional[List[Dict]] = None,
                return_sources: bool = True,
                temperature: float = 0.0) -> Dict:
        """
        질문에 대한 답변 생성
        
        Args:
            query: 사용자 질문
            contexts: 검색된 문서 텍스트 리스트
            sources: 문서 메타데이터 리스트
            return_sources: 출처 정보 포함 여부
            temperature: LLM temperature (0=결정적, 1=창의적)
        
        Returns:
            {
                'answer': 생성된 답변,
                'sources': 참조 출처 (선택적),
                'contexts_used': 사용된 문서 수
            }
        """
        if not self.has_api_key:
            return self._generate_fallback(query, contexts, sources)
        
        # 프롬프트 생성
        prompt = self._build_prompt(query, contexts, sources)
        
        try:
            # LLM 호출 (OpenAI 1.0.0+ API)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 대한민국 법률 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            result = {
                'answer': answer,
                'contexts_used': len(contexts),
                'model': self.model
            }
            
            if return_sources and sources:
                result['sources'] = self._format_sources(sources)
            
            return result
        
        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            return self._generate_fallback(query, contexts, sources)
    
    def _build_prompt(self, query: str, contexts: List[str], 
                     sources: Optional[List[Dict]] = None) -> str:
        """프롬프트 생성"""
        
        # 문서 포맷팅
        formatted_contexts = []
        for i, context in enumerate(contexts, 1):
            source_info = ""
            if sources and i-1 < len(sources):
                src = sources[i-1]
                book_id = src.get('book_id', 'Unknown')
                category = src.get('category', 'Unknown')
                source_info = f" [{book_id}, {category}]"
            
            # 너무 긴 문서는 잘라내기
            if len(context) > 1500:
                context = context[:1500] + "..."
            
            formatted_contexts.append(f"[문서 {i}]{source_info}\n{context}")
        
        contexts_text = "\n\n".join(formatted_contexts)
        
        # 프롬프트 생성
        prompt = self.PROMPT_TEMPLATE.format(
            contexts=contexts_text,
            query=query
        )
        
        return prompt
    
    def _format_sources(self, sources: List[Dict]) -> List[str]:
        """출처 정보 포맷팅"""
        
        formatted = []
        for src in sources:
            book_id = src.get('book_id', 'Unknown')
            category = src.get('category', '')
            year = src.get('publication_year', '')
            
            source_str = f"{book_id}"
            if category:
                source_str += f" ({category})"
            if year:
                source_str += f" - {year}년"
            
            formatted.append(source_str)
        
        return formatted
    
    def _generate_fallback(self, query: str, contexts: List[str], 
                          sources: Optional[List[Dict]] = None) -> Dict:
        """API 키가 없을 때 폴백 응답"""
        
        # 단순히 검색된 문서의 일부를 반환
        answer = "검색 결과:\n\n"
        for i, context in enumerate(contexts[:3], 1):
            preview = context[:200] + "..." if len(context) > 200 else context
            answer += f"{i}. {preview}\n\n"
        
        answer += "\n[참고] OpenAI API 키가 설정되지 않아 검색 결과만 표시됩니다."
        
        result = {
            'answer': answer,
            'contexts_used': len(contexts),
            'model': 'fallback'
        }
        
        if sources:
            result['sources'] = self._format_sources(sources)
        
        return result


# 사용 예시
if __name__ == '__main__':
    generator = LegalGenerator(model='gpt-3.5-turbo')
    
    query = "부동산 매매계약 합의해제의 효력은?"
    contexts = [
        "매매계약의 합의해제로 인하여 매수인 앞으로 이전되었던 소유권은 당연히 매도인에게 원상태로 복귀된다.",
        "합의해제의 경우 계약은 소급적으로 소멸한다."
    ]
    sources = [
        {'book_id': 'LJU000001', 'category': '민사소송법', 'publication_year': 2015},
        {'book_id': 'LJU000002', 'category': '민법', 'publication_year': 2018}
    ]
    
    result = generator.generate(query, contexts, sources)
    
    print("질문:", query)
    print("\n답변:")
    print(result['answer'])
    
    if 'sources' in result:
        print("\n참조 출처:")
        for src in result['sources']:
            print(f"  - {src}")

