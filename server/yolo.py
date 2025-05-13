import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import tempfile # 임시 파일 생성을 위해 추가
import os # 파일 경로 처리를 위해 추가

# 현재 디렉토리를 기준으로 모듈을 임포트하도록 수정
from . import gemini
from .websocket_manager import connection_manager, StatusMessage

# YOLO 모델 로드 (애플리케이션 시작 시 한 번 로드 권장)
model = YOLO('./best.pt') # 예시 모델 경로
print("YOLO model loaded.") # 로딩 확인 로그

# 최근 3개 프레임의 화재 관련 객체 감지 여부 저장 (True: 감지됨, False: 감지 안됨)
recent_detections = deque(maxlen=3)

# 화재 의심 신고 로직
async def fire_alert(current_frame: np.ndarray):
    """
    화재 의심 상황 시 호출될 함수.
    Gemini API로 프레임 분석 후 웹소켓으로 알림.
    """
    print("화재 의심! 연속 3프레임에서 객체 감지됨. Gemini API로 검증 시작...")

    temp_image_path = ""
    try:
        # 1. 현재 프레임을 임시 이미지 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix="fire_frame_") as temp_file:
            cv2.imwrite(temp_file.name, current_frame)
            temp_image_path = temp_file.name
        
        print(f"임시 이미지 저장: {temp_image_path}")

        # 2. Gemini API 호출 (동기 함수이므로 asyncio.to_thread 사용)
        # gemini.use_gemini가 동기 함수라고 가정합니다. 만약 비동기 함수라면 await gemini.use_gemini(...)
        gemini_response = await asyncio.to_thread(gemini.use_gemini, temp_image_path)
        print(f"Gemini API 응답: {gemini_response}")

        # 3. Gemini API 응답 분석 (간단한 키워드 검색)
        # 실제로는 더 정교한 분석 로직이 필요할 수 있습니다.
        if "화재" in gemini_response or "연기" in gemini_response or "불" in gemini_response or "dangerous" in gemini_response.lower():
            print("Gemini API 결과: 화재 또는 위험 상황으로 판단됨.")
            description = f"화재 발생 의심 (Gemini 분석: {gemini_response[:200]})" # 메시지 길이 제한
            status_message = StatusMessage(status="dangerous", description=description)
            
            # 4. 웹소켓으로 브로드캐스트
            await connection_manager.broadcast_json(status_message)
            print("웹소켓으로 화재 경보 브로드캐스트 완료.")
        else:
            print("Gemini API 결과: 화재 또는 명확한 위험 상황으로 판단되지 않음.")
            # 일반적인 위험 상황이 아닌 경우, 'hazardous' 또는 'normal' 상태로 브로드캐스트 할 수도 있습니다.
            # 여기서는 위험으로 판단될 때만 보내도록 합니다.
            # description = f"객체 감지 (Gemini 분석: {gemini_response[:200]})"
            # status_message = StatusMessage(status="hazardous", description=description) # 예시
            # await connection_manager.broadcast_json(status_message)


    except Exception as e:
        print(f"fire_alert 처리 중 오류 발생: {e}")
    finally:
        # 임시 파일 삭제
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"임시 이미지 삭제: {temp_image_path}")
            except Exception as e:
                print(f"임시 이미지 삭제 중 오류: {e}")

async def process_frame_with_yolo(frame: np.ndarray) -> np.ndarray:
    """
    주어진 NumPy 배열 형태의 프레임에 대해 YOLO 객체 검출을 수행하고,
    객체가 마킹된 프레임을 반환합니다. 연속 3프레임 화재 감지 시 fire_alert 호출.
    """
    results = model.predict(frame, verbose=False)
    current_frame_detected = False
    marked_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                if cls in [0, 1, 2]: # 화재 관련 클래스 (예시)
                    current_frame_detected = True

                if hasattr(model, 'names') and cls < len(model.names):
                    label = f"{model.names[cls]} {conf:.2f}"
                else:
                    label = f"Class {cls} {conf:.2f}"
                
                cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(marked_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing box: {e}")
                continue

    recent_detections.append(current_frame_detected)

    if len(recent_detections) == 3 and all(recent_detections):
        # fire_alert를 호출할 때 현재 프레임(마킹된 프레임 또는 원본 프레임)을 전달합니다.
        # 원본 프레임을 보내는 것이 분석에 더 적합할 수 있습니다.
        await fire_alert(frame) # 원본 프레임(frame)을 전달

    await asyncio.sleep(0.001)
    return marked_frame