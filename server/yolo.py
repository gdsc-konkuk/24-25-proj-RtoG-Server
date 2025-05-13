# server/yolo.py
# 화재 감지 및 이벤트 처리 로직
# - YOLO 모델을 사용한 화재 감지
# - 이벤트 영상 저장 및 썸네일 생성
# - Gemini API를 통한 화재 검증

import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import tempfile 
import os 
import time
from datetime import datetime
from sqlalchemy.orm import Session
from database import get_db

import gemini
from websocket_manager import connection_manager, StatusMessage
from config import settings
from models import FireEvent, Video

# YOLO 모델 로드 (애플리케이션 시작 시 한 번 로드 권장)
model = YOLO('./best.pt') 
print("YOLO model loaded.")

# 최근 5개 프레임의 화재 관련 객체 감지 여부 저장 (True: 감지됨, False: 감지 안됨)
recent_detections = deque(maxlen=5)

# 마지막 Gemini API 호출 시간 추적
last_gemini_call_time = 0
GEMINI_COOLDOWN = 30  # 30초 쿨다운

def save_event_video(video_path: str, frame_number: int, fps: float, db: Session, video_id: str, analysis: str, event_type: str) -> tuple[str, str]:
    """
    의심 프레임을 중심으로 앞뒤 5초 영상을 저장하고 썸네일을 생성합니다.
    
    Args:
        video_path: 원본 영상 경로
        frame_number: 의심 프레임 번호
        fps: 영상의 초당 프레임 수
        db: 데이터베이스 세션
        video_id: 비디오 ID
        analysis: Gemini 분석 결과
        event_type: 이벤트 타입
        
    Returns:
        tuple[str, str]: (저장된 영상 경로, 썸네일 경로)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("영상을 열 수 없습니다.")
            
        # 영상 정보 가져오기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 저장할 프레임 범위 계산 (앞뒤 5초)
        frames_to_save = int(fps * 5)  # 5초에 해당하는 프레임 수
        start_frame = max(0, frame_number - frames_to_save)
        end_frame = min(total_frames, frame_number + frames_to_save)
        
        # 저장 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_dir = os.path.join(settings.RECORD_STORAGE_PATH, "events")
        os.makedirs(event_dir, exist_ok=True)
        
        # 파일명 생성
        video_filename = f"event_{timestamp}.mp4"
        thumb_filename = f"event_{timestamp}_thumb.jpg"
        
        # 영상 저장
        output_path = os.path.join(event_dir, video_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 의심 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 프레임 저장
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
        out.release()
        
        # 썸네일 생성
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, thumbnail_frame = cap.read()
        if ret:
            thumbnail_path = os.path.join(event_dir, thumb_filename)
            cv2.imwrite(thumbnail_path, thumbnail_frame)
        else:
            thumbnail_path = None
            
        cap.release()
        
        # DB에 이벤트 저장 (상대 경로 사용)
        relative_video_path = os.path.join(settings.RECORD_STORAGE_PATH, "events", video_filename)
        relative_thumb_path = os.path.join(settings.RECORD_STORAGE_PATH, "events", thumb_filename) if thumbnail_path else None
        
        event = FireEvent(
            video_id=video_id,
            event_type=event_type,
            analysis=analysis,
            file_path=relative_video_path,
            thumbnail_path=relative_thumb_path
        )
        db.add(event)
        db.commit()
        
        return relative_video_path, relative_thumb_path
        
    except Exception as e:
        print(f"Error saving event video: {e}")
        if 'out' in locals():
            out.release()
        if 'cap' in locals():
            cap.release()
        raise

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

    # 임시 이미지 파일로 저장
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_image_path = temp_file.name
        cv2.imwrite(temp_image_path, current_frame)

    try:
        # Gemini API로 분석
        analysis = gemini.use_gemini(temp_image_path)
        status = analysis.get("status", "normal")  # 기본값을 normal로 변경
        
        # 이벤트 타입 설정 (gemini.py의 상태값 사용)
        event_type = status  # dangerous, normal, hazardous 중 하나
        
        # 웹소켓으로 알림 먼저 전송
        await connection_manager.broadcast_json(
            StatusMessage(
                status=event_type,
                description=analysis.get("analysis", ""),
                cctvId=cctv_id
            )
        )
        
        # DB 세션 생성
        db = next(get_db())
        try:
            # 영상 저장
            video_path = f"{settings.VIDEO_STORAGE_PATH}/{cctv_id}.mp4"
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.release()
            
            saved_video_path, thumbnail_path = save_event_video(
                video_path=video_path,
                frame_number=current_frame_number,
                fps=fps,
                db=db,
                video_id=cctv_id,
                analysis=analysis.get("analysis", ""),
                event_type=event_type
            )
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error in fire_alert: {e}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

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