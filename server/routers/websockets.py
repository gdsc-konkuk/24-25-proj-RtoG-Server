# server/routers/websockets.py
# WebSocket 통신 엔드포인트 정의
# - 용의자/특이사항 실시간 알림 (/ws/suspects)
# - 비디오 실시간 스트리밍 (/ws/stream/{video_id})

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from websocket_manager import connection_manager
from models import Video

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
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"Client disconnected from video {video_id}: {websocket.client}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket)

    except Exception as e:
        print(f"Error in video stream {video_id} for client {websocket.client}: {e}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket) 