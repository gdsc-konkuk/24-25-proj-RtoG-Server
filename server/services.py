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
from models import Video # 모델 및 스키마 import (필요시)
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
        """Detect fire in a frame using YOLO model. (기존 루트 main.py의 기능)"""
        results = self.yolo_model(frame, verbose=False)  # verbose=False to suppress output
        
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    # 클래스 ID 0이 화재라고 가정 (YOLO 모델 학습에 따라 다름)
                    # 실제 클래스 이름과 ID를 확인해야 합니다.
                    # 예를 들어, model.names 에서 클래스 목록 확인 가능
                    # 여기서는 화재 클래스가 'fire'이고 ID가 특정 값이라고 가정합니다.
                    # class_id = int(box.cls)
                    # if self.yolo_model.names[class_id].lower() == "fire": 
                    #   confidence = float(box.conf)
                    #   return True, confidence
                    
                    # 임시로, 어떤 객체든 감지되면 화재로 간주 (실제로는 클래스 필터링 필요)
                    confidence = float(box.conf)
                    # print(f"Detected object with confidence: {confidence}") # 디버깅용
                    # 실제 화재 클래스 ID로 필터링 해야함. 예시로 첫번째 클래스가 화재라고 가정.
                    if int(box.cls[0]) == 0: # 이 부분은 모델에 따라 수정 필요
                         return True, confidence
        return False, 0.0

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