# server/gemini.py
# Google Gemini Pro Vision API를 사용한 이미지 분석
# - 이미지 파일을 base64로 인코딩
# - API 호출을 통한 이미지 분석
# - 분석 결과 반환 (status, description 딕셔너리 형태)

import base64
import requests
from config import settings
from typing import Dict, Literal

# StatusMessage와 동일한 상태 정의 (의존성을 피하기 위해 여기에 정의)
StatusType = Literal["dangerous", "normal", "hazardous"]

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_gemini_response(response_text: str) -> Dict[str, str]:
    """
    Gemini 응답 텍스트를 파싱하여 status와 description을 추출합니다.
    예상 형식:
    Status: [dangerous|hazardous|normal]
    Description: [상세 설명]
    """
    status: StatusType = "normal" # 기본값
    description = "분석 결과를 파싱할 수 없습니다." # 기본값
    
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.lower().startswith("status:"):
            extracted_status = line.split(":", 1)[1].strip().lower()
            if extracted_status in ["dangerous", "hazardous", "normal"]:
                status = extracted_status
            else:
                 print(f"Warning: Gemini returned unknown status '{extracted_status}'. Defaulting to 'normal'.")
                 status = "normal" # 알 수 없는 상태면 normal로 처리
        elif line.lower().startswith("description:"):
            description = line.split(":", 1)[1].strip()
            
    # 만약 Status, Description 형식이 아니라면, 응답 텍스트 전체를 description으로 사용하고 status는 기본값 유지
    if description == "분석 결과를 파싱할 수 없습니다." and response_text:
        description = response_text # 전체 응답을 설명으로 사용
        # 키워드 기반으로 status 재설정 시도 (옵션)
        if "화재" in description or "연기" in description or "불" in description:
             status = "dangerous"
        elif "위험" in description:
             status = "hazardous"


    return {"status": status, "description": description}


def use_gemini(image_path: str) -> Dict[str, str]:
    """
    Gemini API를 호출하여 이미지를 분석하고, 상태(status)와 설명(description)을 포함한 딕셔너리를 반환합니다.
    """
    default_error_response = {"status": "normal", "description": "이미지 분석 중 오류 발생"}
    if not settings.GEMINI_API_KEY:
        return {"status": "normal", "description": "Gemini API key not configured"}

    try:
        base64_image = encode_image_to_base64(image_path)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # 프롬프트 수정: 상태 분류 및 설명 요청 (설명은 한국어로), 형식 지정
        prompt = ('''Analyze this image for fire, smoke, or other hazards. 
Classify the situation as 'dangerous', 'hazardous', or 'normal'. 
Provide a concise description **in Korean** of what you see. 
Respond ONLY in the following format:
Status: [classification]
Description: [description in Korean]''')

        data = {
            "contents": [{
                "parts":[{
                    "text": prompt
                }, {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                }]
            }],
             # 추가 파라미터 설정 (선택 사항)
             "generationConfig": {
                "temperature": 0.2, # 낮은 온도로 설정하여 일관된 형식 유도
                "maxOutputTokens": 100 # 응답 길이 제한
            }
        }

        response = requests.post(
            f"{API_URL}?key={settings.GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=20 # 타임아웃 증가
        )
        
        if response.status_code == 200:
            response_data = response.json()
            try:
                if "candidates" in response_data and response_data["candidates"]:
                     content = response_data["candidates"][0].get("content", {})
                     if "parts" in content and content["parts"]:
                         response_text = content["parts"][0].get("text", "")
                         if response_text:
                              return parse_gemini_response(response_text)
                         else:
                              return {"status": "normal", "description": "Gemini 응답에서 텍스트를 찾을 수 없습니다."}
                     else:
                         return {"status": "normal", "description": "Gemini 응답 형식이 예상과 다릅니다. (parts 없음)"}
                else:
                     error_info = response_data.get("promptFeedback", {}).get("blockReason", "Unknown reason")
                     return {"status": "normal", "description": f"Gemini API가 유효한 응답을 반환하지 않았습니다. 이유: {error_info}"}

            except (IndexError, KeyError, TypeError) as e:
                 print(f"Error parsing Gemini response structure: {e}")
                 error_desc = f"Gemini 응답 파싱 오류: {e}. 응답: {str(response_data)[:200]}"
                 return {"status": "normal", "description": error_desc}

        else:
            error_body = response.text
            return {"status": "normal", "description": f"API 호출 실패: {response.status_code}. 응답: {error_body[:200]}"}

    except requests.exceptions.Timeout:
         return {"status": "normal", "description": "Gemini API 호출 시간 초과"}
    except Exception as e:
        return {"status": "normal", "description": f"이미지 분석 중 오류 발생: {str(e)}"}