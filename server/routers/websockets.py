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
from typing import Tuple, List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from websocket_manager import connection_manager
from models import Video
from config import config

model = YOLO(config.YOLO_MODEL_PATH)

# websocket으로 들어온 이미지 처리하는 파이프라인
async def process_frame(
    frame_data: Dict[str, Any],
    video_id: str,
    owner_id: str,
    websocket: WebSocket
) -> bool:
    # 필수 필드 확인
    if not all(key in frame_data for key in ['stream_id', 'timestamp', 'frame_type', 'frame_data']):
        print("Missing required fields in received data")
        return True
        
    if frame_data['stream_id'] != video_id:
        print(f"Stream ID mismatch: expected {video_id}, got {frame_data['stream_id']}")
        return True
        
    if frame_data['frame_type'] != 'jpeg':
        print(f"Unsupported frame type: {frame_data['frame_type']}")
        return True
    
    # 이미지 디코딩
    try:
        img = decode_base64_image(frame_data['frame_data'])
        if img is None:
            return True
    except Exception as e:
        print(f"Error decoding image: {e}")
        return True
    
    # YOLO 객체 감지 수행
    detections, fire_detected = process_yolo_detection(img)
    
    # 화재 감지 시 알림 전송
    if fire_detected:
        await send_fire_alert(video_id, owner_id, frame_data['timestamp'])
    
    # 결과를 클라이언트에게 전송
    await websocket.send_json({
        'status': 'success',
        'detections': detections,
        'fire_detected': fire_detected,
        'timestamp': frame_data['timestamp']
    })
    
    return True

# base64 이미지 디코딩
def decode_base64_image(base64_data: str) -> np.ndarray:
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

# YOLO 모델을 사용하여 이미지에서 객체를 감지하는 함수
def process_yolo_detection(img: np.ndarray) -> Tuple[List[Dict[str, Any]], bool]:
    results = model(img)
    detections = []
    fire_detected = False
    
    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            
            detection = {
                'class': class_name,
                'confidence': confidence,
                'bbox': box.xyxy[0].tolist()
            }
            detections.append(detection)
            
            if class_name == 'fire' and confidence > 0.5:
                fire_detected = True
    
    return detections, fire_detected

# 화재 감지 알림을 전송하는 함수
async def send_fire_alert(video_id: str, owner_id: str, timestamp: str):
    pass



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