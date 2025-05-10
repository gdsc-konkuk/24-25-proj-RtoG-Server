import google.generativeai as genai
from dotenv import load_dotenv
import os
import sys
from PIL import Image

# .env 파일 로드
load_dotenv()

# API 키 설정
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

genai.configure(api_key=GOOGLE_API_KEY)

def use_gemini(image_path: str) -> str:
    """
    Gemini API를 사용하여 이미지를 분석합니다.
    
    Args:
        image_path (str): 분석할 이미지의 경로
        
    Returns:
        str: 분석 결과 (위험/주의/안전)
    """
    try:
        # 이미지 로드
        image = Image.open(image_path)
        
        # Gemini 모델 설정
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 이미지 분석 요청 (간단하고 명확한 프롬프트)
        response = model.generate_content([
            "이 이미지에서 화재가 발생했는지 확인하고, 다음 세 가지 중 하나로만 응답해주세요:\n"
            "1. '위험': 큰 화재가 발생했거나 긴급한 상황\n"
            "2. '주의': 작은 화재가 발생했거나 잠재적 위험\n"
            "3. '안전': 화재가 없거나 위험하지 않은 상황",
            image
        ])
        
        # 응답 정리
        result = response.text.strip().lower()
        if '위험' in result:
            return '위험'
        elif '주의' in result:
            return '주의'
        else:
            return '안전'
            
    except Exception as e:
        print(f"Gemini API 오류: {str(e)}")
        return "안전"  # 오류 발생 시 안전으로 처리

if __name__ == "__main__":
    use_gemini()