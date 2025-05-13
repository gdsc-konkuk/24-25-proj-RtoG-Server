# server/gemini.py
# Google Gemini Pro Vision API를 사용한 이미지 분석
# - 이미지 파일을 base64로 인코딩
# - API 호출을 통한 이미지 분석
# - 분석 결과 반환 (status, description 딕셔너리 형태)

import google.generativeai as genai
from PIL import Image
from config import settings
from typing import Dict, Literal, Set

# StatusMessage와 동일한 상태 정의 (의존성을 피하기 위해 여기에 정의)
StatusType = Literal["dangerous", "normal", "hazardous"]

# 상수 정의
DEFAULT_DESCRIPTION = "Couldn't parse the response."
DANGEROUS_KEYWORDS: Set[str] = {"fire", "smoke", "flame", "burning", "blaze"}
HAZARDOUS_KEYWORDS: Set[str] = {"danger", "hazard", "risk", "warning", "caution"}

def parse_gemini_response(response_text: str) -> Dict[str, str]:
    """
    Gemini 응답 텍스트를 파싱하여 status와 description을 추출합니다.
    예상 형식:
    Status: [dangerous|hazardous|normal]
    Description: [상세 설명]
    """
    status: StatusType = "normal"  # 기본값
    description = DEFAULT_DESCRIPTION  # 기본값
    
    if not response_text:
        return {"status": status, "description": description}
    
    # 응답 텍스트를 소문자로 변환하여 키워드 매칭을 용이하게 함
    response_lower = response_text.lower()
    
    # Status와 Description 형식으로 파싱 시도
    lines = response_text.strip().split('\n')
    for line in lines:
        line_lower = line.lower()
        if line_lower.startswith("status:"):
            extracted_status = line.split(":", 1)[1].strip().lower()
            if extracted_status in ["dangerous", "hazardous", "normal"]:
                status = extracted_status
            else:
                print(f"Warning: Gemini returned unknown status '{extracted_status}'. Defaulting to 'normal'.")
        elif line_lower.startswith("description:"):
            description = line.split(":", 1)[1].strip()
    
    # 형식화된 응답이 없는 경우, 전체 응답을 description으로 사용하고 키워드 기반으로 status 결정
    if description == DEFAULT_DESCRIPTION:
        description = response_text.strip()
        # 키워드 기반 status 결정
        if any(keyword in response_lower for keyword in DANGEROUS_KEYWORDS):
            status = "dangerous"
        elif any(keyword in response_lower for keyword in HAZARDOUS_KEYWORDS):
            status = "hazardous"
    
    return {"status": status, "description": description}

def use_gemini(image_path: str) -> Dict[str, str]:
    """
    Gemini API를 호출하여 이미지를 분석하고, 상태(status)와 설명(description)을 포함한 딕셔너리를 반환합니다.
    """
    if not settings.GEMINI_API_KEY:
        return {"status": "normal", "description": "Gemini API key not configured"}

    try:
        # Gemini API 설정
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # 이미지 로드
        image = Image.open(image_path)
        
        # Gemini 모델 설정
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 프롬프트 설정
        prompt = '''Analyze this image for fire, smoke, or other hazards. 
Classify the situation as 'dangerous', 'hazardous', or 'normal'. 
Provide a concise description of what you see. 
Respond ONLY in the following format:
Status: [classification]
Description: [description]'''

        # 이미지 분석 요청
        response = model.generate_content([prompt, image])
        
        if response.text:
            return parse_gemini_response(response.text)
        else:
            return {"status": "normal", "description": "Empty gemini response."}

    except Exception as e:
        error_message = f"Error while processing image: {str(e)}"
        print(error_message)
        return {"status": "normal", "description": error_message}