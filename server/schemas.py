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

class LiveCCTV(BaseModel):
    """실시간 CCTV 정보"""
    id: str
    name: str
    location: str
    socket_id: str

    model_config = ConfigDict(from_attributes=True)

class LiveResponse(BaseModel):
    """실시간 CCTV 목록 응답"""
    cctvs: list[LiveCCTV]

class EventDetail(BaseModel):
    """화재 이벤트 상세 정보"""
    eventId: str
    cctv_name: str
    location: str
    timestamp: datetime
    video_url: str
    description: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class EventSummary(BaseModel):
    """화재 이벤트 요약 정보 (썸네일 포함)"""
    eventId: str
    cctv_name: str
    location: str
    thumbnail_url: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

class DailyEvents(BaseModel):
    """일자별 화재 이벤트 그룹"""
    date: str
    events: list[EventSummary]

class RecordsResponse(BaseModel):
    """화재 이벤트 기록 목록 응답"""
    records: list[DailyEvents] 