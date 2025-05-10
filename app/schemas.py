from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class VideoBase(BaseModel):
    id: str
    filename: str
    location: str
    cctv_name: str
    installation_date: str
    resolution: str
    angle: str
    status: str

class VideoCreate(VideoBase):
    pass

class Video(VideoBase):
    class Config:
        from_attributes = True

class FireEventBase(BaseModel):
    video_id: str
    timestamp: datetime
    event_type: str
    confidence: float
    analysis: str

class FireEventCreate(FireEventBase):
    pass

class FireEvent(FireEventBase):
    id: str

    class Config:
        from_attributes = True

class WebSocketMessage(BaseModel):
    action: str
    stream_type: Optional[str] = None 