# server/schemas.py
# API 요청/응답 데이터의 유효성 검사와 직렬화를 위한 Pydantic 모델 정의

from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime

class Video(BaseModel):
    id: str
    filename: str
    location: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    cctv_name: Optional[str] = None
    installation_date: Optional[str] = None
    resolution: Optional[str] = None
    angle: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class FireEvent(BaseModel):
    id: int
    video_id: str
    timestamp: datetime
    event_type: str
    confidence: Optional[float] = None
    analysis: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class SuspectEvent(BaseModel):
    id: int
    video_id: str
    timestamp: datetime
    confidence: float
    analysis: str

    model_config = ConfigDict(from_attributes=True) 