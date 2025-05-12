# server/websockets.py
# 이 파일은 WebSocket 연결 관리 및 실시간 메시지 처리를 위한 UnifiedConnectionManager 클래스를 정의합니다.
# 이 매니저는 다음과 같은 기능을 수행합니다:
# - 클라이언트 WebSocket 연결 수락 및 관리 (connect, disconnect)
# - 모든 연결된 클라이언트에게 JSON 메시지 브로드캐스팅 (broadcast_json)

from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Set, Optional, Any
import asyncio
import json

class UnifiedConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        print("Initializing UnifiedConnectionManager")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected: {websocket.client}. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket disconnected: {websocket.client}. Total: {len(self.active_connections)}")
        else:
            print(f"Attempted to disconnect inactive client: {websocket.client}")

    async def broadcast_json(self, message: Dict[str, Any]):
        """Sends a JSON message to all connected clients."""
        disconnected_clients = []
        print(f"Broadcasting message to {len(self.active_connections)} clients: {message}")
        connections_to_broadcast = list(self.active_connections)
        for connection in connections_to_broadcast:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                print(f"Client disconnected during broadcast: {connection.client}")
                disconnected_clients.append(connection)
            except Exception as e:
                print(f"Error broadcasting to {connection.client}: {e}")
                pass

        for client in disconnected_clients:
            self.disconnect(client)

connection_manager = UnifiedConnectionManager()
