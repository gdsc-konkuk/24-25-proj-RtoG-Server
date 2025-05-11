# server/routers/websockets.py
# WebSocket 통신 엔드포인트 정의
# - 용의자/특이사항 실시간 알림 (/ws/suspects)
# - 비디오 실시간 스트리밍 (/ws/stream/{video_id})

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import asyncio
import json
from typing import Dict, Any

from database import get_db
from websocket_manager import connection_manager
from models import Video
from services import video_processing_service

router = APIRouter()

@router.websocket("/ws/suspects")
async def suspects_endpoint(
    websocket: WebSocket,
    owner_id: str = Query(..., description="CCTV 소유자 ID")
):
    """용의자/특이사항 실시간 알림을 위한 WebSocket 엔드포인트"""
    # 1. WebSocket 연결 수립
    await connection_manager.connect(websocket)
    
    # 2. 용의자 알림 구독
    await connection_manager.subscribe_to_suspects(websocket, owner_id)
    
    try:
        # 3. 메시지 수신 대기
        while True:
            data = await websocket.receive_text()
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        # 4. 연결 해제 처리
        print(f"Suspect client disconnected: {websocket.client}")
        connection_manager.unsubscribe_from_suspects(websocket)
        connection_manager.disconnect(websocket)
    except Exception as e:
        # 5. 에러 처리
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
    """CCTV 영상 스트리밍을 위한 WebSocket 엔드포인트
    :param video_id: Video 모델의 id (CCTV 식별자)
    :param owner_id: CCTV 소유자 ID
    """
    # 1. WebSocket 연결 수립
    await connection_manager.connect(websocket)
    
    # 2. 비디오 존재 여부 및 접근 권한 확인
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        print(f"Stream requested for non-existent video: {video_id}")
        await websocket.close(code=4004, reason=f"Video with id {video_id} not found")
        connection_manager.disconnect(websocket)
        return

    if video.owner_id != owner_id:
        print(f"Unauthorized access attempt to video {video_id} by owner {owner_id}")
        await websocket.close(code=4003, reason="Unauthorized access")
        connection_manager.disconnect(websocket)
        return

    # 3. 비디오 스트림 구독
    print(f"Client {websocket.client} connecting to video stream: {video_id} ({video.cctv_name or 'unnamed'})")
    await connection_manager.subscribe_to_video_stream(websocket, video_id, db)
    
    try:
        # 4. 프레임 처리 루프
        while True:
            # 4.1. 프레임 데이터 수신
            data = await websocket.receive_text()
            
            try:
                # 4.2. JSON 파싱
                json_data = json.loads(data)
                
                # 4.3. 프레임 데이터 검증
                video_processing_service.validate_frame_data(json_data, video_id)
                
                # 4.4. 이미지 디코딩
                img = video_processing_service.decode_base64_image(json_data['frame_data'])
                if img is None:
                    continue
                
                # 4.5. 화재 감지 분석
                fire_detected, confidence, gemini_result = await video_processing_service.analyze_frame_for_fire(img)
                
                # 4.6. 화재 감지시 알림 전송
                if fire_detected and gemini_result:
                    await video_processing_service.send_fire_alert(
                        owner_id=owner_id,
                        video_id=video_id,
                        timestamp=json_data['timestamp'],
                        yolo_confidence=confidence,
                        gemini_description=gemini_result
                    )
                
            except json.JSONDecodeError:
                print("Invalid JSON data received")
                continue
            except ValueError as e:
                print(f"Frame validation error: {e}")
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        # 5. 연결 해제 처리
        print(f"Client disconnected from video {video_id}: {websocket.client}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket)

    except Exception as e:
        # 6. 에러 처리
        print(f"Error in video stream {video_id} for client {websocket.client}: {e}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket) 