# api-server/routers/websockets.py
# 이 파일은 WebSocket을 사용한 실시간 통신 엔드포인트를 정의합니다.
# FastAPI의 APIRouter를 사용하여 WebSocket 관련 경로들을 그룹화하고,
# 클라이언트와의 지속적인 연결을 통해 데이터를 주고받는 로직을 포함합니다.
# 주요 기능:
# - 용의자/특이사항 발생 실시간 알림 WebSocket 엔드포인트 (GET /ws/suspects)
# - 특정 비디오에 대한 실시간 스트리밍 WebSocket 엔드포인트 (GET /ws/stream/{video_id})
# 이러한 엔드포인트들은 UnifiedConnectionManager를 사용하여 WebSocket 연결 관리 및 메시지 브로드캐스팅을 처리합니다.

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.orm import Session
import asyncio

from ..database import get_db
from ..websockets import connection_manager # 통합된 ConnectionManager 사용
from .. import models # 모델 import

router = APIRouter()

@router.websocket("/ws/suspects")
async def suspects_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    await connection_manager.subscribe_to_suspects(websocket)
    try:
        while True:
            # 클라이언트로부터 메시지를 받을 필요는 현재 없음. 알림 전용.
            data = await websocket.receive_text() 
            # 필요하다면 여기서 클라이언트 메시지 처리 로직 추가
            # print(f"Received from suspect client: {data}") 
            await asyncio.sleep(0.01) # CPU 사용 방지를 위한 짧은 sleep
    except WebSocketDisconnect:
        print(f"Suspect client disconnected: {websocket.client}")
        connection_manager.unsubscribe_from_suspects(websocket)
        connection_manager.disconnect(websocket)
    except Exception as e:
        print(f"Error in suspects_endpoint for {websocket.client}: {e}")
        connection_manager.unsubscribe_from_suspects(websocket)
        connection_manager.disconnect(websocket)

@router.websocket("/ws/stream/{video_id}") # video_id를 경로 매개변수로 받음
async def stream_endpoint(
    websocket: WebSocket, 
    video_id: str, 
    db: Session = Depends(get_db)
):
    await connection_manager.connect(websocket)
    
    # video_id를 사용하여 특정 비디오 스트림에 구독
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        print(f"Stream requested for non-existent video_id: {video_id}")
        await websocket.close(code=4004, reason=f"Video with id {video_id} not found.")
        connection_manager.disconnect(websocket)
        return

    print(f"Client {websocket.client} attempting to subscribe to video stream: {video_id}")
    await connection_manager.subscribe_to_video_stream(websocket, video_id, db)
    
    try:
        while True:
            # 클라이언트로부터 스트리밍 제어 메시지를 받을 수 있음 (예: 중지 요청)
            # 현재 ConnectionManager는 서버 주도 스트리밍이므로, 클라이언트 메시지는 로깅만 하거나 간단히 처리
            data = await websocket.receive_text()
            # print(f"Received from stream client {websocket.client} for video {video_id}: {data}")
            # 필요시 여기서 data를 파싱하여 스트리밍 제어 (예: manager.pause_stream(video_id) 등)
            await asyncio.sleep(0.01) # CPU 사용 방지

    except WebSocketDisconnect:
        print(f"Stream client disconnected: {websocket.client} for video {video_id}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket)
        # 만약 특정 video_id에 더 이상 클라이언트가 없으면 스트리밍 중지 로직 호출 고려
        # (UnifiedConnectionManager의 unsubscribe_from_video_stream 내부 또는 여기서 명시적으로)
        # if not connection_manager.streaming_clients.get(video_id):
        #     print(f"No more clients for {video_id}, stopping session.")
        #     await connection_manager.stop_streaming_session(video_id, db)

    except Exception as e:
        print(f"Error in stream_endpoint for {websocket.client}, video {video_id}: {e}")
        await connection_manager.unsubscribe_from_video_stream(websocket, video_id)
        connection_manager.disconnect(websocket) 