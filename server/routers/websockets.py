# server/routers/websockets.py
# WebSocket 통신 엔드포인트 정의
# - 용의자/특이사항 실시간 알림 (/ws/suspects)
# - 비디오 실시간 스트리밍 (/ws/stream/{video_id})

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import asyncio
import base64
import json
import numpy as np
import cv2
from ultralytics import YOLO
import datetime
from typing import Tuple, List, Dict, Any, Optional
import tempfile
import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from websocket_manager import connection_manager
from models import Video
from config import config
from services import video_processing_service, analysis_service

model = YOLO(config.YOLO_MODEL_PATH)

class FrameValidationError(Exception):
    pass

# 프레임 데이터의 유효성을 검증
def validate_frame_data(frame_data: Dict[str, Any], video_id: str) -> None:
    if not all(key in frame_data for key in ['stream_id', 'timestamp', 'frame_type', 'frame_data']):
        raise FrameValidationError("Missing required fields in received data")
        
    if frame_data['stream_id'] != video_id:
        raise FrameValidationError(f"Stream ID mismatch: expected {video_id}, got {frame_data['stream_id']}")
        
    if frame_data['frame_type'] != 'jpeg':
        raise FrameValidationError(f"Unsupported frame type: {frame_data['frame_type']}")

# Base64 인코딩된 이미지를 디코딩
def decode_base64_image(base64_data: str) -> Optional[np.ndarray]:
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

# Gemini API를 사용하여 화재 감지 확인  
async def analyze_fire_with_gemini(img: np.ndarray) -> Tuple[bool, str]:
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        try:
            cv2.imwrite(temp_file.name, img)
            gemini_result = await analysis_service.analyze_image_with_gemini(temp_file.name)
            
            fire_keywords = ['화재', '불', '연기', 'fire', 'smoke', 'burning']
            gemini_confirms_fire = any(keyword in gemini_result.lower() for keyword in fire_keywords)
            
            return gemini_confirms_fire, gemini_result
        finally:
            os.unlink(temp_file.name)

# 화재 감지 알림을 전송
async def send_fire_alert(
    owner_id: str,
    video_id: str,
    timestamp: str,
    yolo_confidence: float,
    gemini_description: str
) -> None:
   pass

# WebSocket으로 들어온 프레임을 처리
async def process_frame(
    frame_data: Dict[str, Any],
    video_id: str,
    owner_id: str,
    websocket: WebSocket
) -> bool:
    try:
        # 프레임 데이터 검증
        validate_frame_data(frame_data, video_id)
        
        # 이미지 디코딩
        img = decode_base64_image(frame_data['frame_data'])
        if img is None:
            return True
            
        # YOLO로 화재 감지
        fire_detected, confidence = video_processing_service.detect_fire_yolo(img)
        
        # YOLO에서 화재 의심되면 Gemini로 2차 확인
        if fire_detected and confidence > 0.5:
            gemini_confirms_fire, gemini_result = await analyze_fire_with_gemini(img)
            
            # Gemini도 화재로 판단하면 알림 전송
            if gemini_confirms_fire:
                await send_fire_alert(
                    owner_id=owner_id,
                    video_id=video_id,
                    timestamp=frame_data['timestamp'],
                    yolo_confidence=confidence,
                    gemini_description=gemini_result
                )
        
    except FrameValidationError as e:
        print(f"Frame validation error: {e}")
        return True
    except Exception as e:
        print(f"Error processing frame: {e}")
        return True
    
    return True

# 라우터 정의
router = APIRouter()

@router.websocket("/ws/suspects")
async def suspects_endpoint(
    websocket: WebSocket,
    owner_id: str = Query(..., description="CCTV 소유자 ID")
):
    await connection_manager.connect(websocket)
    await connection_manager.subscribe_to_suspects(websocket, owner_id)
    try:
        while True:
            data = await websocket.receive_text() 
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        print(f"Suspect client disconnected: {websocket.client}")
        connection_manager.unsubscribe_from_suspects(websocket)
        connection_manager.disconnect(websocket)
    except Exception as e:
        print(f"Error in suspects_endpoint for {websocket.client}: {e}")
        connection_manager.unsubscribe_from_suspects(websocket)
        connection_manager.disconnect(websocket)

@router.websocket("/ws/{video_id}")
async def video_stream_endpoint(
    websocket: WebSocket, 
    video_id: str, 
    owner_id: str = Query(..., description="CCTV 소유자 ID"),
    db: Session = Depends(get_db)
):
    """
    CCTV 영상 스트리밍을 위한 WebSocket 엔드포인트
    :param video_id: Video 모델의 id (CCTV 식별자)
    :param owner_id: CCTV 소유자 ID
    """
    await connection_manager.connect(websocket)
    
    # 비디오 ID가 데이터베이스에 존재하는지 확인
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        print(f"Stream requested for non-existent video: {video_id}")
        await websocket.close(code=4004, reason=f"Video with id {video_id} not found")
        connection_manager.disconnect(websocket)
        return

    # 소유자 확인
    if video.owner_id != owner_id:
        print(f"Unauthorized access attempt to video {video_id} by owner {owner_id}")
        await websocket.close(code=4003, reason="Unauthorized access")
        connection_manager.disconnect(websocket)
        return

    print(f"Client {websocket.client} connecting to video stream: {video_id} ({video.cctv_name or 'unnamed'})")
    await connection_manager.subscribe_to_video_stream(websocket, video_id, db)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                json_data = json.loads(data)
                should_continue = await process_frame(json_data, video_id, owner_id, websocket)
                if not should_continue:
                    break
            except json.JSONDecodeError:
                print("Invalid JSON data received")
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"Client disconnected from video {video_id}: {websocket.client}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket)

    except Exception as e:
        print(f"Error in video stream {video_id} for client {websocket.client}: {e}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket) 