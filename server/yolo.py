import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import tempfile 
import os 
import time

import gemini
from websocket_manager import connection_manager, StatusMessage
from config import settings

# YOLO 모델 로드 (애플리케이션 시작 시 한 번 로드 권장)
model = YOLO('./best.pt') 
print("YOLO model loaded.")

# 최근 5개 프레임의 화재 관련 객체 감지 여부 저장 (True: 감지됨, False: 감지 안됨)
recent_detections = deque(maxlen=5)

# 마지막 Gemini API 호출 시간 추적
last_gemini_call_time = 0
GEMINI_COOLDOWN = 30  # 30초 쿨다운

# 화재 의심 신고 로직
async def fire_alert(current_frame: np.ndarray, cctv_id: str):
    """
    화재 의심 상황 시 호출될 함수.
    Gemini API로 프레임 분석 후 웹소켓으로 알림.
    
    Args:
        current_frame: 현재 프레임 이미지
        cctv_id: CCTV 식별자
    """
    global last_gemini_call_time
    current_time = time.time()
    
    # 마지막 호출 후 30초가 지나지 않았다면 스킵
    if current_time - last_gemini_call_time < GEMINI_COOLDOWN:
        return

    print(f"화재 의심! CCTV {cctv_id}에서 연속 5프레임에서 객체 감지됨. Gemini API로 검증 시작...")
    last_gemini_call_time = current_time

    temp_image_path = ""
    try:
        # 1. 현재 프레임을 임시 이미지 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix="fire_frame_") as temp_file:
            cv2.imwrite(temp_file.name, current_frame)
            temp_image_path = temp_file.name
        
        print(f"임시 이미지 저장: {temp_image_path}")

        # 2. Gemini API 호출 (동기 함수이므로 asyncio.to_thread 사용)
        # gemini.use_gemini가 동기 함수라고 가정합니다. 만약 비동기 함수라면 await gemini.use_gemini(...)
        gemini_result = await asyncio.to_thread(gemini.use_gemini, temp_image_path)
        print(f"Gemini API 결과: {gemini_result}")

        # 3. Gemini API 결과 딕셔너리에서 status와 description 추출
        status = gemini_result.get("status", "normal") # 기본값 normal
        description = gemini_result.get("description", "Gemini 분석 결과 없음")

        # 4. 상태가 'dangerous' 또는 'hazardous'일 경우 웹소켓으로 브로드캐스트
        if status in ["dangerous", "hazardous"]:
            print(f"Gemini API 결과: {status} 상태 감지됨.")
            status_message = StatusMessage(
                status=status, 
                description=description[:200],
                cctvId=cctv_id
            )
            
            # 알림 메시지 형식 출력
            print("\n=== WebSocket 알림 메시지 ===")
            print(f"상태: {status}")
            print(f"CCTV ID: {cctv_id}")
            print(f"설명: {description[:200]}")
            print("===========================\n")
            
            # 웹소켓으로 브로드캐스트
            await connection_manager.broadcast_json(status_message)
            print(f"웹소켓으로 CCTV {cctv_id} 화재 경보 브로드캐스트 완료.")
        else:
            print(f"Gemini API 결과: {status} 상태. 브로드캐스트하지 않음.")

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

async def process_frame_with_yolo(frame: np.ndarray, cctv_id: str) -> np.ndarray:
    """
    주어진 NumPy 배열 형태의 프레임에 대해 YOLO 객체 검출을 수행하고,
    객체가 마킹된 프레임을 반환합니다. 연속 5프레임 화재 감지 시 fire_alert 호출.
    
    Args:
        frame: 처리할 프레임 이미지
        cctv_id: CCTV 식별자
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

    if len(recent_detections) == 5 and all(recent_detections):
        await fire_alert(frame, cctv_id)

    await asyncio.sleep(0.001)
    return marked_frame