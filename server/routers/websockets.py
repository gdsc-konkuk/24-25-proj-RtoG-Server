# server/routers/websockets.py
# WebSocket 통신 엔드포인트 정의
# - 용의자/특이사항 실시간 알림 (/ws/suspects)
# - 비디오 실시간 스트리밍 (/ws/stream/{video_id})

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
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
async def suspects_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    await connection_manager.subscribe_to_suspects(websocket)
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

@router.websocket("/ws/stream/{video_id}")
async def stream_endpoint(
    websocket: WebSocket, 
    video_id: str, 
    db: Session = Depends(get_db)
):
    await connection_manager.connect(websocket)
    
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        print(f"Stream requested for non-existent video_id: {video_id}")
        await websocket.close(code=4004, reason=f"Video with id {video_id} not found.")
        connection_manager.disconnect(websocket)
        return

    print(f"Client {websocket.client} attempting to subscribe to video stream: {video_id}")
    await connection_manager.subscribe_to_video_stream(websocket, video_id, db)
    
    try:
        while True:
            data = await websocket.receive_text()
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"Stream client disconnected: {websocket.client} for video {video_id}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket)

    except Exception as e:
        print(f"Error in stream_endpoint for {websocket.client}, video {video_id}: {e}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket) 