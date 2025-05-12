# server/services.py
# 이 파일은 애플리케이션의 주요 비즈니스 로직을 서비스 계층의 클래스들로 분리하여 정의합니다.
# 각 서비스 클래스는 특정 도메인 관련 기능을 캡슐화하여 재사용성과 테스트 용이성을 높입니다.
# 주요 서비스:
# - VideoProcessingService: 비디오 파일 저장, 비디오 프레임에서 특정 객체의 좌표 추출,
#   YOLO 모델을 사용한 프레임 내 화재 감지 등의 기능을 제공합니다.
# - AnalysisService: 이미지 분석 관련 로직을 담당하며, 현재는 gemini.py의
#   Gemini API 호출 함수를 사용하여 이미지 분석 기능을 제공합니다.
# - StreamingService는 제거되었습니다.

import os
import cv2
import numpy as np
from typing import List, Tuple, Deque, Optional, Dict, Any
import asyncio
import collections
import base64
from ultralytics import YOLO
from datetime import datetime, timedelta
import time # 추가

from config import settings # 설정 import
from database import get_db, Base # 데이터베이스 관련 import (필요시)
from models import Video, FireEvent # 모델 및 스키마 import (필요시)
from gemini import use_gemini as call_gemini_api

# websocket_manager 임포트 추가
from websocket_manager import connection_manager # send_fire_alert에서 사용

# YOLO 모델 로드 - AnalysisService 또는 VideoService 초기화 시로 이동 고려
# model_path = settings.YOLO_MODEL_PATH
# yolo_model = YOLO(model_path)
# yolo_model.conf = settings.YOLO_CONFIDENCE_THRESHOLD

# Gemini API 호출 함수 (gemini.py에서 가져오거나 직접 구현)
# 여기서는 gemini.py의 함수를 직접 호출한다고 가정하고, 필요시 해당 함수를 여기에 포함시킬 수 있음

class VideoProcessingService:
    """기존 app/services/video_service.py 의 VideoService 역할과 
       루트 main.py의 비디오 파일 처리 관련 기능을 통합"""
    def __init__(self):
        self.upload_dir = settings.VIDEO_UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
        # YOLO 모델은 AnalysisService 또는 여기서 로드할 수 있습니다.
        self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
        self.yolo_model.conf = settings.YOLO_CONFIDENCE_THRESHOLD

    def save_video(self, file: Any) -> str: # FastAPI의 UploadFile 타입 사용 예정
        """Save uploaded video file and return the path."""
        # Ensure filename is safe and unique if necessary
        file_path = os.path.join(self.upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = file.file.read() # await file.read() for async
            buffer.write(content)
        return file_path

    def process_video_extract_coordinates(self, video_path: str) -> List[Tuple[float, float, float]]:
        """Process video and return list of (timestamp, x, y) coordinates. (기존 app의 기능)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        coordinates = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: # fps가 0이면 기본값 사용 또는 에러 처리
            fps = 30 # 예시 기본값
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    timestamp = frame_count / fps
                    coordinates.append((timestamp, cx, cy))

            frame_count += 1

        cap.release()
        return coordinates

    def detect_fire_yolo(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect fire in a frame using YOLO model. (기존 루트 main.py의 기능)
        
        Returns:
            tuple[bool, float]: (화재 감지 여부, 신뢰도)
            화재로 간주되는 클래스:
            - 0: 흑색연기
            - 1: 백색/회색연기
            - 2: 화염
        """
        results = self.yolo_model(frame, verbose=False)  # verbose=False to suppress output
        
        max_confidence = 0.0
        fire_detected = False
        
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf)
                    
                    # 클래스 ID 0(흑색연기), 1(백색/회색연기), 2(화염)를 화재로 간주
                    if class_id in [0, 1, 2]:
                        if confidence > max_confidence:
                            max_confidence = confidence
                            fire_detected = True
        
        return fire_detected, max_confidence

    def validate_frame_data(self, frame_data: Dict[str, Any], video_id: str) -> None:
        """프레임 데이터의 유효성을 검증"""
        if not all(key in frame_data for key in ['stream_id', 'timestamp', 'frame_type', 'frame_data']):
            raise ValueError("Missing required fields in received data")
            
        if frame_data['stream_id'] != video_id:
            raise ValueError(f"Stream ID mismatch: expected {video_id}, got {frame_data['stream_id']}")
            
        if frame_data['frame_type'] != 'jpeg':
            raise ValueError(f"Unsupported frame type: {frame_data['frame_type']}")

    def decode_base64_image(self, base64_data: str) -> Optional[np.ndarray]:
        """Base64 인코딩된 이미지를 디코딩"""
        try:
            img_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            return img
        except Exception as e:
            print(f"Error in decode_base64_image: {e}")
            return None

    async def analyze_frame_for_fire(self, img: np.ndarray) -> Tuple[bool, float, Optional[str]]:
        """프레임에서 화재를 감지하고 필요시 Gemini로 확인"""
        # YOLO로 화재 감지
        fire_detected, confidence = self.detect_fire_yolo(img)
        
        # YOLO에서 화재 의심되면 Gemini로 2차 확인
        if fire_detected and confidence > 0.5:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                try:
                    cv2.imwrite(temp_file.name, img)
                    gemini_result = await analysis_service.analyze_image_with_gemini(temp_file.name)
                    
                    fire_keywords = ['화재', '불', '연기', 'fire', 'smoke', 'burning']
                    gemini_confirms_fire = any(keyword in gemini_result.lower() for keyword in fire_keywords)
                    
                    if gemini_confirms_fire:
                        return True, confidence, gemini_result
                finally:
                    os.unlink(temp_file.name)
        
        return fire_detected, confidence, None

    async def send_fire_alert(
        self,
        owner_id: str,
        video_id: str,
        timestamp: str,
        yolo_confidence: float,
        gemini_description: Optional[str] = None
    ) -> None:
        """화재 감지 알림을 모든 연결된 클라이언트에게 전송"""
        alert_message = {
            "type": "fire_alert",
            "data": {
                "video_id": video_id,
                "timestamp": timestamp,
                "yolo_confidence": round(yolo_confidence, 2),
                "description": gemini_description if gemini_description else "화재 의심 상황이 감지되었습니다.",
            }
        }
        await connection_manager.broadcast_json(alert_message)
        print(f"Fire alert sent for video {video_id}: {alert_message['data']}")

class AnalysisService:
    """Gemini API 호출 등 분석 관련 서비스"""
    def __init__(self):
        # API 키는 config에서 로드되어 gemini.py 모듈에 설정되어 있음
        pass

    async def analyze_image_with_gemini(self, image_path: str) -> str:
        """Gemini API를 사용하여 이미지 분석"""
        try:
            return call_gemini_api(image_path)
        except Exception as e:
            print(f"Gemini analysis error in service: {e}")
            return "이미지 분석 중 오류가 발생했습니다."

class LiveService:
    """실시간 CCTV 스트리밍 가능 목록 제공 (웹소켓 연결 정보는 더 이상 유효하지 않을 수 있음)"""

    @staticmethod
    def get_lives(db) -> list[dict]:
        """
        데이터베이스에서 비디오 목록을 가져와 반환합니다.
        socket_id 필드는 더 이상 WebSocket 연결과 직접 관련되지 않을 수 있습니다.
        """
        videos = db.query(Video).all()
        return [
            {
                "id": f"cctv_{v.id:03d}",
                "name": v.cctv_name or v.filename,
                "address": v.location or "",
                "video_id": v.id
            }
            for v in videos
        ]

class RecordService:
    """화재 이벤트 기록 관련 서비스"""
    
    @staticmethod
    def get_records(db, start=None, end=None):
        """
        일자별로 그룹화된 화재 이벤트 목록을 반환합니다.
        
        Args:
            db: 데이터베이스 세션
            start: 조회 시작 날짜 (YYYY-MM-DD)
            end: 조회 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            list[dict]: 일자별로 그룹화된 이벤트 목록
            [
                {
                    "date": "2024-03-15",
                    "events": [
                        {
                            "eventId": "evt_001",
                            "cctv_name": "강릉시청 앞 CCTV-1",
                            "address": "강원도 강릉시",
                            "thumbnail_url": "/static/events/evt_001.jpg",
                            "timestamp": "2024-03-15T14:23:00"
                        }
                    ]
                }
            ]
        """
        try:
            query = db.query(FireEvent).join(Video)
            
            if start:
                try:
                    start_date = datetime.strptime(start, "%Y-%m-%d")
                    query = query.filter(FireEvent.timestamp >= start_date)
                except ValueError:
                    raise ValueError("Invalid start date format. Use YYYY-MM-DD.")
            if end:
                try:
                    end_date = datetime.strptime(end, "%Y-%m-%d")
                    query = query.filter(FireEvent.timestamp < end_date + timedelta(days=1))
                except ValueError:
                    raise ValueError("Invalid end date format. Use YYYY-MM-DD.")
                
            events = query.order_by(FireEvent.timestamp.desc()).all()
            
            # 일자별로 그룹화
            daily_events = {}
            for event in events:
                date_str = event.timestamp.strftime("%Y-%m-%d")
                if date_str not in daily_events:
                    daily_events[date_str] = []
                    
                # 썸네일/비디오 URL 생성 로직 개선 필요 (실제 파일 경로 기반)
                # 예시: /static/event_media/{event.id}/thumbnail.jpg
                #      /static/event_media/{event.id}/video.mp4
                # 아래는 임시 형식
                thumbnail_base_path = f"/static/events/evt_{event.id:03d}" # 예시 경로

                daily_events[date_str].append({
                    "eventId": f"evt_{event.id:03d}",
                    "cctv_name": event.video.cctv_name or event.video.filename,
                    "address": event.video.location or "주소 정보 없음", # 기본값 제공
                    "timestamp": event.timestamp.isoformat() # ISO 8601 형식
                })
                
            # 날짜순으로 정렬하여 반환
            return [
                {"date": date, "events": event_list}
                for date, event_list in sorted(daily_events.items(), reverse=True)
            ]
        except ValueError as e:
            raise ValueError(f"Invalid date format: {str(e)}")
        except Exception as e:
            print(f"Error fetching records: {e}")
            raise Exception("Error fetching records")

    @staticmethod
    def get_record_detail(db, event_id):
        """
        특정 화재 이벤트의 상세 정보를 반환합니다.
        
        Args:
            db: 데이터베이스 세션
            event_id: 이벤트 ID (evt_001 형식)
            
        Returns:
            dict: 이벤트 상세 정보
            {
                "eventId": "evt_001",
                "cctv_name": "강릉시청 앞 CCTV-1",
                "address": "강원도 강릉시",
                "timestamp": "2024-03-15T14:23:00",
                "video_url": "/static/events/evt_001.mp4",
                "description": "화재 감지 이벤트 상세 설명"
            }
            
        Raises:
            ValueError: 잘못된 이벤트 ID 형식
            Exception: 데이터베이스 조회 오류
        """
        try:
            # event_id 형식 검증 (예: "evt_001")
            if not event_id.startswith("evt_") or not event_id[4:].isdigit():
                 raise ValueError("Invalid event ID format. Expected 'evt_XXX'.")
            numeric_id = int(event_id[4:])

            event = db.query(FireEvent).join(Video).filter(FireEvent.id == numeric_id).first()
            if not event:
                return None
                
            # 비디오 URL 생성 로직 개선 필요
            # video_base_path = f"/static/events/evt_{event.id:03d}" # 예시 경로

            return {
                "eventId": f"evt_{event.id:03d}",
                "cctv_name": event.video.cctv_name or event.video.filename,
                "address": event.video.location or "주소 정보 없음",
                "timestamp": event.timestamp.isoformat(),
                "description": event.analysis or "상세 분석 정보 없음" # 기본값 제공
            }
        except ValueError as e: # 형식 오류 처리
            raise ValueError(str(e)) # 오류 메시지 그대로 전달
        except Exception as e:
            print(f"Error fetching record detail for {event_id}: {e}")
            raise Exception("Error fetching record detail") 