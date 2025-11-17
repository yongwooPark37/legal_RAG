"""
Generator (LLM) 평가 시스템

LLM-as-a-Judge를 사용한 생성 품질 평가
"""

from typing import List
import openai
import os


class GeneratorEvaluator:
    """
    생성 품질 평가 (LLM-as-a-Judge)
    
    평가 지표:
    1. Faithfulness (충실성): 답변이 문맥에 기반하는가?
    2. Relevancy (관련성): 답변이 질문에 정확히 대답하는가?
    3. Completeness (완전성): 답변이 충분한가?
    """
    
    def __init__(self, judge_model='gpt-4'):
        self.judge_model = judge_model
        # API 키 설정 (환경 변수에서)
        openai.api_key = os.getenv('OPENAI_API_KEY', '')
    
    def evaluate_faithfulness(self, query: str, contexts: List[str], answer: str) -> Dict:
        """
        충실성 평가: 답변이 제공된 문맥에만 기반하는가?
        
        환각(Hallucination) 탐지가 핵심
        
        Returns:
            {'score': 4.5, 'reason': '...', 'has_hallucination': False}
        """
        
        contexts_text = "\n\n".join([f"[문서 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""다음을 평가하세요:

<문맥 (제공된 법률 문서)>
{contexts_text}
</문맥>

<사용자 질문>
{query}
</사용자 질문>

<AI의 답변>
{answer}
</AI의 답변>

평가 기준:
1. 답변이 제공된 <문맥>에만 기반하고 있는가?
2. 문맥에 없는 정보를 지어내지 않았는가? (환각/Hallucination)
3. 문맥을 정확히 인용하거나 요약했는가?

점수: 1-5
- 1점: 문맥과 전혀 무관하거나 대부분 환각
- 2점: 일부만 문맥 기반, 많은 환각 포함
- 3점: 대체로 문맥 기반이나 일부 부정확
- 4점: 거의 완벽하게 문맥 기반, 약간의 해석
- 5점: 완벽하게 문맥만 사용, 환각 없음

출력 형식:
점수: [1-5]
환각 여부: [예/아니오]
이유: [1-2문장 설명]
"""
        
        response = self._call_judge_llm(prompt)
        parsed = self._parse_evaluation_response(response)
        
        return parsed
    
    def evaluate_relevancy(self, query: str, answer: str) -> Dict:
        """
        관련성 평가: 답변이 질문에 정확히 대답하는가?
        """
        
        prompt = f"""다음을 평가하세요:

<사용자 질문>
{query}
</사용자 질문>

<AI의 답변>
{answer}
</AI의 답변>

평가 기준:
1. 답변이 질문의 핵심에 정확히 대답하는가?
2. 질문과 무관한 내용은 없는가?
3. 답변이 질문의 의도를 제대로 파악했는가?

점수: 1-5
- 1점: 질문과 전혀 무관
- 2점: 질문과 약간 관련 있으나 핵심 놓침
- 3점: 질문에 부분적으로 대답
- 4점: 질문에 거의 완벽하게 대답
- 5점: 질문의 모든 측면에 완벽하게 대답

출력 형식:
점수: [1-5]
이유: [1-2문장 설명]
"""
        
        response = self._call_judge_llm(prompt)
        parsed = self._parse_evaluation_response(response)
        
        return parsed
    
    def evaluate_completeness(self, query: str, answer: str, contexts: List[str]) -> Dict:
        """
        완전성 평가: 답변이 충분히 상세한가?
        """
        
        contexts_text = "\n\n".join([f"[문서 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""다음을 평가하세요:

<문맥>
{contexts_text}
</문맥>

<질문>
{query}
</질문>

<답변>
{answer}
</답변>

평가 기준:
1. 문맥에 있는 관련 정보를 충분히 활용했는가?
2. 질문에 답하기에 충분한 세부사항을 제공했는가?
3. 중요한 법률 근거나 출처를 명시했는가?

점수: 1-5
- 1점: 너무 짧거나 불충분
- 3점: 기본적인 답변이나 세부사항 부족
- 5점: 충분히 상세하고 완전한 답변

출력 형식:
점수: [1-5]
이유: [설명]
"""
        
        response = self._call_judge_llm(prompt)
        parsed = self._parse_evaluation_response(response)
        
        return parsed
    
    def evaluate_all(self, query: str, contexts: List[str], answer: str) -> Dict:
        """전체 평가 수행"""
        
        return {
            'faithfulness': self.evaluate_faithfulness(query, contexts, answer),
            'relevancy': self.evaluate_relevancy(query, answer),
            'completeness': self.evaluate_completeness(query, answer, contexts)
        }
    
    def _call_judge_llm(self, prompt: str) -> str:
        """판단용 LLM 호출"""
        
        if not openai.api_key:
            # API 키가 없으면 더미 응답 (테스트용)
            return "점수: 4\n환각 여부: 아니오\n이유: 테스트 응답입니다."
        
        try:
            response = openai.ChatCompletion.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "당신은 법률 AI 시스템의 답변을 평가하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0  # 일관성을 위해 낮게
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"LLM 호출 실패: {e}")
            return "점수: 0\n이유: 평가 실패"
    
    def _parse_evaluation_response(self, response: str) -> Dict:
        """평가 응답 파싱"""
        
        result = {
            'score': 0,
            'reason': '',
            'has_hallucination': False,
            'raw_response': response
        }
        
        # 점수 추출
        import re
        score_match = re.search(r'점수:\s*(\d+)', response)
        if score_match:
            result['score'] = int(score_match.group(1))
        
        # 이유 추출
        reason_match = re.search(r'이유:\s*(.+?)(?:\n|$)', response, re.DOTALL)
        if reason_match:
            result['reason'] = reason_match.group(1).strip()
        
        # 환각 여부
        if '환각' in response or 'hallucination' in response.lower():
            hallucination_match = re.search(r'환각 여부:\s*(.+?)(?:\n|$)', response)
            if hallucination_match:
                result['has_hallucination'] = '예' in hallucination_match.group(1) or 'yes' in hallucination_match.group(1).lower()
        
        return result


# 사용 예시
if __name__ == '__main__':
    evaluator = GeneratorEvaluator()
    
    query = "부동산 매매계약 합의해제의 효력은?"
    contexts = ["매매계약의 합의해제로 인하여 매수인 앞으로 이전되었던 소유권은 당연히 매도인에게 원상태로 복귀된다."]
    answer = "합의해제로 인해 소유권은 매도인에게 원상 복귀됩니다. 근거: 대법원 판례"
    
    results = evaluator.evaluate_all(query, contexts, answer)
    
    print("\n평가 결과:")
    for metric, result in results.items():
        print(f"\n{metric}:")
        print(f"  점수: {result['score']}/5")
        print(f"  이유: {result['reason']}")

