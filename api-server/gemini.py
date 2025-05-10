# api-server/gemini.py
# 이 파일은 Google Gemini API와의 연동 로직을 포함합니다.
# config.py에서 Gemini API 키를 로드하여 API 클라이언트를 설정하고,
# 이미지 경로를 입력받아 Gemini 멀티모달 모델을 사용하여 이미지 분석을 수행하는
# use_gemini 함수를 제공합니다. 이 함수는 이미지 내 화재 위험도를 분석하여
# '위험', '주의', '안전' 중 하나의 문자열로 결과를 반환합니다.

import google.generativeai as genai
# from dotenv import load_dotenv # 변경: BaseSettings가 처리
import os
import sys
from PIL import Image
from .config import settings # 변경: 설정 파일 import

# .env 파일 로드 # 변경: BaseSettings가 처리
# load_dotenv()

# API 키 설정
# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") # 변경: 설정 파일에서 가져옴
if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았거나 config에 없습니다.")

genai.configure(api_key=settings.GEMINI_API_KEY)

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
    # 이 부분은 실제 API 호출 시에는 직접 실행되지 않으므로, 테스트 코드로 분리하거나 삭제 고려
    # 여기서는 config에서 키를 잘 불러오는지 확인하는 용도로 남겨둘 수 있지만, 
    # 실제 운영시는 삭제하거나 if 문으로 감싸는 것이 좋음
    if settings.GEMINI_API_KEY:
        print("Gemini API 키가 설정되었습니다. 테스트를 진행하려면 이미지를 제공해야 합니다.")
        # 예: use_gemini("test_image.jpg")
    else:
        print("Gemini API 키가 설정되지 않아 테스트를 진행할 수 없습니다.")