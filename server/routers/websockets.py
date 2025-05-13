# server/routers/websockets.py
# WebSocket 통신 엔드포인트 정의
# - 클라이언트 연결 유지 및 서버 발신 메시지 수신 (/ws)

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from typing import Dict, Any

from websocket_manager import connection_manager

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """단순 연결 유지 및 서버 발신 메시지 수신을 위한 WebSocket 엔드포인트"""
    # 1. WebSocket 연결 수립 및 관리자 등록
    await connection_manager.connect(websocket)
    
    try:
        # 2. 연결 유지 (클라이언트로부터 메시지를 받을 필요는 없지만, 연결 종료 감지를 위해 필요)
        while True:
            # 클라이언트로부터 데이터를 기다리지만, 실제 데이터 처리는 하지 않음
            # receive_text() 또는 receive_bytes()를 호출해야 연결 종료를 감지할 수 있음
            data = await websocket.receive_text() 
            # 필요하다면 여기서 수신된 데이터 로깅 또는 간단한 처리 가능
            # print(f"Received data from {websocket.client}: {data}") 
            await asyncio.sleep(0.01) # CPU 사용 방지
    except WebSocketDisconnect:
        # 3. 연결 해제 처리
        print(f"Client disconnected: {websocket.client}")
        connection_manager.disconnect(websocket)
    except Exception as e:
        # 4. 기타 에러 처리
        print(f"Error in websocket_endpoint for {websocket.client}: {e}")
        # 에러 발생 시에도 연결 해제 처리
        connection_manager.disconnect(websocket) 