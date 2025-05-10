# server/gemini.py
# Google Gemini Pro Vision API를 사용한 이미지 분석
# - 이미지 파일을 base64로 인코딩
# - API 호출을 통한 이미지 분석
# - 분석 결과 반환

import base64
import requests
import json
from config import settings

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def use_gemini(image_path: str) -> str:
    if not settings.GEMINI_API_KEY:
        return "Gemini API key not configured"

    try:
        base64_image = encode_image_to_base64(image_path)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts":[{
                    "text": "이 이미지에서 화재나 연기, 위험한 상황이 보이나요? 보인다면 어떤 위험이 있는지 자세히 설명해주세요."
                }, {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                }]
            }]
        }

        response = requests.post(
            f"{API_URL}?key={settings.GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            response_data = response.json()
            if "candidates" in response_data:
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
            return "분석 결과를 찾을 수 없습니다."
        else:
            return f"API 호출 실패: {response.status_code}"

    except Exception as e:
        return f"이미지 분석 중 오류 발생: {str(e)}"

if __name__ == "__main__":
    # 이 부분은 실제 API 호출 시에는 직접 실행되지 않으므로, 테스트 코드로 분리하거나 삭제 고려
    # 여기서는 config에서 키를 잘 불러오는지 확인하는 용도로 남겨둘 수 있지만, 
    # 실제 운영시는 삭제하거나 if 문으로 감싸는 것이 좋음
    if settings.GEMINI_API_KEY:
        print("Gemini API 키가 설정되었습니다. 테스트를 진행하려면 이미지를 제공해야 합니다.")
        # 예: use_gemini("test_image.jpg")
    else:
        print("Gemini API 키가 설정되지 않아 테스트를 진행할 수 없습니다.")