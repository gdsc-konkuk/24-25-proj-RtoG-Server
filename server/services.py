# server/services.py
# 이 파일은 애플리케이션의 주요 비즈니스 로직을 서비스 계층의 클래스들로 분리하여 정의합니다.
# 각 서비스 클래스는 특정 도메인 관련 기능을 캡슐화하여 재사용성과 테스트 용이성을 높입니다.
# 주요 서비스:
# - VideoProcessingService: 비디오 파일 저장, 비디오 프레임에서 특정 객체의 좌표 추출,
#   YOLO 모델을 사용한 프레임 내 화재 감지 등의 기능을 제공합니다.
# - AnalysisService: 이미지 분석 관련 로직을 담당하며, 현재는 gemini.py의
#   Gemini API 호출 함수를 사용하여 이미지 분석 기능을 제공합니다.
# - StreamingService: 특정 비디오 파일에 대한 실시간 프레임 스트리밍을 관리합니다.
#   비디오 캡처, 프레임 버퍼링, 프레임 인코딩 및 제공 기능을 포함합니다.

import os
import cv2
import numpy as np
from typing import List, Tuple, Deque, Optional, Dict, Set, Any
import asyncio
import collections
import base64
from ultralytics import YOLO
from datetime import datetime
import time # 추가

from config import settings # 설정 import
from database import get_db, Base # 데이터베이스 관련 import (필요시)
from models import Video, FireEvent # 모델 및 스키마 import (필요시)
from gemini import use_gemini as call_gemini_api

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
        gemini_description: str
    ) -> None:
        """화재 감지 알림을 전송"""
        # TODO: 실제 알림 전송 로직 구현
        print(f"Fire alert for video {video_id}: {gemini_description} (conf: {yolo_confidence})")
        pass

class AnalysisService:
    """Gemini API 호출 등 분석 관련 서비스"""
    def __init__(self):
        # API 키는 config에서 로드되어 gemini.py 모듈에 설정되어 있음
        pass

    async def analyze_image_with_gemini(self, image_path: str) -> str:
        """Gemini API를 사용하여 이미지 분석 (비동기 Wrapper 예시)"""
        # call_gemini_api는 동기 함수이므로, 실제 비동기 환경에서는 
        # asyncio.to_thread 등을 사용하거나 gemini 라이브러리가 비동기를 지원하는지 확인 필요.
        # 여기서는 간단히 호출하는 형태로 둠.
        # loop = asyncio.get_event_loop()
        # return await loop.run_in_executor(None, call_gemini_api, image_path)
        try:
            return call_gemini_api(image_path)
        except Exception as e:
            print(f"Gemini analysis error in service: {e}")
            return "오류" # 또는 다른 기본값

class StreamingService:
    """기존 루트 main.py의 VideoStream 클래스와 스트리밍 관련 로직 담당"""
    def __init__(self, video_id: str, video_path: str):
        self.video_id = video_id
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.fps = 5  # 기본 FPS
        self.buffer_size = self.fps * 5  # 5초 버퍼
        self.frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=self.buffer_size)
        self._open_capture()

    def _open_capture(self):
        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found at {self.video_path}")
            self.is_running = False
            return
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video stream for {self.video_id} at {self.video_path}")
            self.is_running = False
        else:
            self.is_running = True
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if actual_fps > 0:
                self.fps = actual_fps # 실제 FPS 사용 또는 고정 FPS 유지 선택
            print(f"Video stream initialized: {self.video_id}, path: {self.video_path}, FPS: {self.fps}")

    async def get_frame(self) -> tuple[bool, str]:
        if not self.is_running or self.cap is None:
            return False, ''
        
        ret, frame = self.cap.read()
        if not ret:
            # 비디오 끝에 도달하면 처음으로 되돌림 (루프 재생)
            print(f"Video {self.video_id} reached end, restarting.")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                print(f"Failed to restart video: {self.video_id}")
                self.is_running = False
                return False, ''
        
        # 프레임 리사이즈 (필요시)
        frame_resized = cv2.resize(frame, (640, 480))
        self.frame_buffer.append(frame_resized.copy())
        
        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, encoded_image = cv2.imencode('.jpg', frame_resized, encode_param)
        if not success:
            print(f"Frame encoding failed for {self.video_id}")
            return False, ''
        
        frame_base64 = base64.b64encode(encoded_image).decode('utf-8')
        return True, frame_base64

    def get_buffered_frames(self) -> List[np.ndarray]:
        return list(self.frame_buffer)

    def release(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        print(f"Video stream terminated: {self.video_id}")

# 서비스 인스턴스 생성 (싱글톤처럼 사용하거나, Depends로 주입)
video_processing_service = VideoProcessingService()
analysis_service = AnalysisService()
# StreamingService는 video_id, video_path 마다 인스턴스화 필요 

class LiveService:
    @staticmethod
    def get_lives(db) -> list[dict]:
        """
        비디오 목록을 반환 (이름, 설치 위치(location), id만 포함)
        프론트엔드에서 Live 탭 진입 시 호출
        """
        videos = db.query(Video).all()
        result = []
        for v in videos:
            result.append({
                "name": getattr(v, "cctv_name", None) or v.filename,
                "address": v.location or "",
                "socketId": v.id
            })
        return result 

class RecordService:
    @staticmethod
    def get_records(db, start=None, end=None):
        query = db.query(FireEvent)
        if start:
            query = query.filter(FireEvent.timestamp >= start)
        if end:
            query = query.filter(FireEvent.timestamp <= end)
        events = query.all()
        grouped = {}
        for event in events:
            date_str = event.timestamp.strftime("%Y-%m-%d")
            if date_str not in grouped:
                grouped[date_str] = []
            grouped[date_str].append({
                "eventId": event.id,
                "cctv_name": event.video.cctv_name,
                "location": event.video.location,
                "thumbnail_url": f"/static/suspects/{event.id}.jpg",
                "video_url": f"/static/suspects/{event.id}.mp4",
                "timestamp": event.timestamp.isoformat()
            })
        result = [{"date": date, "events": evts} for date, evts in sorted(grouped.items(), reverse=True)]
        return result

    @staticmethod
    def get_record_detail(db, event_id):
        event = db.query(FireEvent).filter(FireEvent.id == event_id).first()
        if not event:
            return None
        return {
            "eventId": event.id,
            "cctv_name": event.video.cctv_name,
            "location": event.video.location,
            "timestamp": event.timestamp.isoformat(),
            "video_url": f"/static/suspects/{event.id}.mp4",
            "description": event.analysis
        } 