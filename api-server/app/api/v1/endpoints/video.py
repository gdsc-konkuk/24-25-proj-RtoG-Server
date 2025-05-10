from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, Depends
from typing import List, Tuple
from ....services.video_service import video_service
from ....websockets.connection_manager import manager
from ....core.config import settings
import uuid

router = APIRouter()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        file_path = video_service.save_video(file)
        return {"message": "Video uploaded successfully", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/process/{file_path:path}")
async def process_video(file_path: str):
    try:
        coordinates = video_service.process_video(file_path)
        return {"coordinates": coordinates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time video processing updates.
    
    Parameters:
    - client_id: A unique identifier for the client connection
    
    Message Format (Client to Server):
    ```json
    {
        "type": "control",
        "action": "start|stop|pause",
        "data": {}
    }
    ```
    
    Response Format (Server to Client):
    ```json
    {
        "type": "update",
        "status": "processing|completed|error",
        "data": {
            "progress": 0.75,
            "message": "Processing frame 150 of 200"
        }
    }
    ```
    
    Example Usage:
    1. Connect to the WebSocket endpoint with a unique client_id
    2. Send control messages to start/stop/pause processing
    3. Receive real-time updates about the processing status
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message({"message": data}, client_id)
    except Exception as e:
        manager.disconnect(websocket, client_id) 