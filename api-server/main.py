from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Set, Optional
import json
import asyncio
from datetime import datetime
import os
import cv2
import numpy as np
from pathlib import Path
import sys
import base64
from ultralytics import YOLO
import collections
from glob import glob

# Add gemini.py path to system path
sys.path.append(str(Path(__file__).parent.parent))
from gemini import use_gemini

# Change relative imports to absolute imports
import models
import schemas
from database import engine, get_db, Base

# Update the model path
model_path = "training/yolo11n.pt"

# Load YOLO model
model = YOLO(model_path)
model.conf = 0.1  # Set confidence threshold (10%)

# Create database tables
print("Creating database tables...")
Base.metadata.drop_all(bind=engine)  # Drop existing tables
Base.metadata.create_all(bind=engine)  # Create new tables
print("Database tables created successfully")

app = FastAPI(
    title="Fire Detection API",
    description="""
    API server for fire detection and video streaming.
    
    ## Features
    * Real-time video streaming
    * Fire detection using YOLO
    * Risk analysis using Gemini
    * WebSocket support for real-time updates
    
    ## WebSocket Endpoints
    * `/ws/suspects` - For fire detection notifications
    * `/ws/stream` - For video streaming
    
    ## Authentication
    Currently, the API does not require authentication.
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

# CORS settings for API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class VideoStream:
    def __init__(self, video_id: str, video_path: str):
        self.video_id = video_id
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        self.is_running = True
        self.fps = 5  # 5FPS standard
        self.buffer_size = self.fps * 5  # 5 seconds of frames
        self.frame_buffer = collections.deque(maxlen=self.buffer_size)
        print(f"Video stream initialized: {video_id}, path: {video_path}")

    async def get_frame(self) -> tuple[bool, str]:
        if not self.is_running:
            return False, ''
        
        success, frame = self.cap.read()
        if not success:
            print(f"Failed to read video frame: {self.video_id}")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()
            if not success:
                print(f"Failed to restart video: {self.video_id}")
                self.is_running = False
                return False, ''
        
        frame = cv2.resize(frame, (640, 480))
        self.frame_buffer.append(frame.copy())
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            print(f"Frame encoding failed: {self.video_id}")
            return False, ''
        
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return True, frame_base64

    def get_buffer_frames(self):
        """Return recent 5 seconds of frames"""
        return list(self.frame_buffer)

    def release(self):
        self.is_running = False
        self.cap.release()
        print(f"Video stream terminated: {self.video_id}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.video_streams: Dict[str, VideoStream] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.streaming_clients: Dict[str, Set[WebSocket]] = {}
        self.analysis_tasks: Dict[str, asyncio.Task] = {}
        self.suspect_clients: Set[WebSocket] = set()
        self.last_suspect_save_time: Dict[str, float] = {}
        self.last_alert_time: Dict[str, float] = {}
        self.last_gemini_time: Dict[str, float] = {}

    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            print(f"WebSocket connected. Current connections: {len(self.active_connections)}")
        except Exception as e:
            print(f"WebSocket connection failed: {str(e)}")

    def disconnect(self, websocket: WebSocket):
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            if websocket in self.suspect_clients:
                self.suspect_clients.remove(websocket)
            for clients in self.streaming_clients.values():
                if websocket in clients:
                    clients.remove(websocket)
            print(f"WebSocket disconnected. Current connections: {len(self.active_connections)}")
        except Exception as e:
            print(f"WebSocket disconnection failed: {str(e)}")

    async def subscribe_to_suspects(self, websocket: WebSocket):
        try:
            self.suspect_clients.add(websocket)
            print(f"Fire detection subscription added. Current subscribers: {len(self.suspect_clients)}")
        except Exception as e:
            print(f"Fire detection subscription failed: {str(e)}")

    async def unsubscribe_from_suspects(self, websocket: WebSocket):
        try:
            self.suspect_clients.discard(websocket)
            print(f"Fire detection subscription removed. Current subscribers: {len(self.suspect_clients)}")
        except Exception as e:
            print(f"Fire detection unsubscription failed: {str(e)}")

    async def notify_suspects(self, video_id: str, timestamp: str, confidence: float, analysis: str, frame: str):
        disconnected_clients = set()
        for client in self.suspect_clients:
            try:
                await client.send_json({
                    "event": "fire_detected",
                    "data": {
                        "id": video_id,
                        "timestamp": timestamp,
                        "confidence": confidence,
                        "analysis": analysis,
                        "frame": frame
                    }
                })
            except Exception as e:
                print(f"Notification failed: {str(e)}")
                disconnected_clients.add(client)
        
        for client in disconnected_clients:
            self.suspect_clients.discard(client)

    async def start_streaming(self, video_id: str, video_path: str):
        try:
            if video_id in self.video_streams:
                print(f"Stream already running: {video_id}")
                return
            
            print(f"Starting new video stream: {video_id}")
            stream = VideoStream(video_id, video_path)
            self.video_streams[video_id] = stream
            self.streaming_clients[video_id] = set()
            
            task = asyncio.create_task(self.stream_video(video_id))
            self.streaming_tasks[video_id] = task
            print(f"Streaming task created: {video_id}")
            
            analysis_task = asyncio.create_task(self.analyze_video(video_id, video_path))
            self.analysis_tasks[video_id] = analysis_task
            print(f"Analysis task created: {video_id}")
        except Exception as e:
            print(f"Error starting stream: {video_id}, error: {str(e)}")
            if video_id in self.video_streams:
                self.video_streams[video_id].release()
                del self.video_streams[video_id]
            raise

    async def analyze_video(self, video_id: str, video_path: str):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        db = next(get_db())
        
        stream = self.video_streams.get(video_id)
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                if frame_count % 25 == 0:  # Check every 5 seconds at 5FPS
                    is_fire, confidence = detect_fire(frame)
                    
                    if is_fire and confidence >= 0.1:
                        import time
                        now = time.time()
                        last_alert = self.last_alert_time.get(video_id, 0)
                        
                        if now - last_alert < 10:
                            print(f"[{video_id}] 10-second cooldown: Skipping Gemini/alert")
                        else:
                            try:
                                temp_frame_path = f"temp_frame_{video_id}.jpg"
                                cv2.imwrite(temp_frame_path, frame)
                                risk_level = use_gemini(temp_frame_path)
                                os.remove(temp_frame_path)
                                print(f"[{video_id}] Gemini analysis result: {risk_level}")
                                
                                if risk_level == '위험' or risk_level == '주의':
                                    self.last_alert_time[video_id] = now
                                    frame_base64 = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
                                    await self.notify_suspects(
                                        video_id=video_id,
                                        timestamp=datetime.now().isoformat(),
                                        confidence=confidence,
                                        analysis=risk_level,
                                        frame=frame_base64
                                    )
                                    
                                    last_save = self.last_suspect_save_time.get(video_id, 0)
                                    if now - last_save >= 30:
                                        self.last_suspect_save_time[video_id] = now
                                        suspect_event = models.SuspectEvent(
                                            video_id=video_id,
                                            timestamp=datetime.now(),
                                            confidence=confidence,
                                            analysis=risk_level
                                        )
                                        db.add(suspect_event)
                                        db.commit()
                                        print(f"[{video_id}] Suspect event saved to database")
                            except Exception as e:
                                print(f"[{video_id}] Error in analysis: {str(e)}")
                
                frame_count += 1
                await asyncio.sleep(0.2)  # 5FPS
                
        except Exception as e:
            print(f"Error in video analysis: {video_id}, error: {str(e)}")
        finally:
            cap.release()

    async def stream_video(self, video_id: str):
        try:
            stream = self.video_streams.get(video_id)
            if not stream:
                print(f"Stream not found: {video_id}")
                return
            
            while stream.is_running:
                success, frame_base64 = await stream.get_frame()
                if not success:
                    print(f"Failed to get frame: {video_id}")
                    break
                
                disconnected_clients = set()
                for client in self.streaming_clients.get(video_id, set()):
                    try:
                        await client.send_json({
                            "event": "frame",
                            "data": {
                                "frame": frame_base64,
                                "timestamp": datetime.now().isoformat()
                            }
                        })
                    except Exception as e:
                        print(f"Failed to send frame: {str(e)}")
                        disconnected_clients.add(client)
                
                for client in disconnected_clients:
                    self.streaming_clients[video_id].discard(client)
                
                await asyncio.sleep(0.2)  # 5FPS
                
        except Exception as e:
            print(f"Error in video streaming: {video_id}, error: {str(e)}")
        finally:
            if video_id in self.video_streams:
                self.video_streams[video_id].release()
                del self.video_streams[video_id]

    async def subscribe_to_video(self, websocket: WebSocket, video_id: str):
        if video_id not in self.streaming_clients:
            self.streaming_clients[video_id] = set()
        self.streaming_clients[video_id].add(websocket)
        print(f"Client subscribed to video: {video_id}")

def detect_fire(frame: np.ndarray) -> tuple[bool, float]:
    results = model(frame)
    for result in results:
        if len(result.boxes) > 0:
            return True, float(result.boxes[0].conf)
    return False, 0.0

# API Endpoints
@app.get(
    "/api/videos",
    response_model=List[schemas.Video],
    summary="Get all videos",
    description="Retrieve a list of all available videos in the system.",
    tags=["Videos"]
)
async def get_videos(db: Session = Depends(get_db)):
    """
    Get all videos.
    
    Returns:
        List[Video]: A list of all videos in the system.
    """
    videos = db.query(models.Video).all()
    return videos

@app.get(
    "/api/suspects/{id}/events",
    response_model=List[schemas.SuspectEvent],
    summary="Get suspect events for a video",
    description="Retrieve all suspect events (fire detections) for a specific video.",
    tags=["Events"]
)
async def get_suspect_events(
    id: str,
    db: Session = Depends(get_db)
):
    """
    Get suspect events for a specific video.
    
    Args:
        id (str): The ID of the video to get events for.
        
    Returns:
        List[SuspectEvent]: A list of suspect events for the specified video.
    """
    events = db.query(models.SuspectEvent).filter(models.SuspectEvent.video_id == id).all()
    return events

# WebSocket Endpoints
@app.websocket("/ws/suspects")
async def suspects_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for fire detection notifications.
    
    Messages:
        - Send "unsubscribe" to stop receiving notifications
        
    Events:
        - "fire_detected": When a fire is detected
            {
                "event": "fire_detected",
                "data": {
                    "id": "string",
                    "timestamp": "string",
                    "confidence": "float",
                    "analysis": "string",
                    "frame": "string" // base64 encoded image
                }
            }
    """
    await manager.connect(websocket)
    try:
        await manager.subscribe_to_suspects(websocket)
        while True:
            data = await websocket.receive_text()
            if data == "unsubscribe":
                await manager.unsubscribe_from_suspects(websocket)
                break
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error in suspects WebSocket: {str(e)}")
        manager.disconnect(websocket)

@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for video streaming.
    
    Initial Message (Required):
        {
            "video_id": "string",
            "video_path": "string"
        }
        
    Stream Format:
        {
            "event": "frame",
            "data": {
                "frame": "string", // base64 encoded image
                "timestamp": "string"
            }
        }
        
    Control:
        - Send "unsubscribe" to stop receiving video stream
    """
    await manager.connect(websocket)
    try:
        data = await websocket.receive_json()
        video_id = data.get("video_id")
        video_path = data.get("video_path")
        
        if not video_id or not video_path:
            await websocket.close(code=1008, reason="Missing video_id or video_path")
            return
        
        await manager.start_streaming(video_id, video_path)
        await manager.subscribe_to_video(websocket, video_id)
        
        while True:
            data = await websocket.receive_text()
            if data == "unsubscribe":
                break
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error in stream WebSocket: {str(e)}")
        manager.disconnect(websocket)

# Initialize connection manager
manager = ConnectionManager() 