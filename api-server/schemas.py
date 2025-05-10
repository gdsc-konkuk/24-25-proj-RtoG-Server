# api-server/schemas.py
# 이 파일은 API 요청 및 응답 데이터의 유효성 검사와 직렬화/역직렬화를 위해 Pydantic 모델(스키마)을 정의합니다.
# 각 스키마는 데이터베이스 모델과 상호작용하거나 API의 데이터 형태를 명확히 하는 데 사용됩니다.
# 주요 스키마:
# - Video: 비디오 정보에 대한 스키마 (ID, 파일명, 상태 등)
# - FireEvent: 화재 이벤트 정보에 대한 스키마 (ID, 비디오 ID, 타임스탬프 등)
# - SuspectEvent: 용의자/특이사항 이벤트 정보에 대한 스키마 (ID, 비디오 ID, 타임스탬프 등)
# 각 스키마 내 Config 클래스의 orm_mode = True 설정은 SQLAlchemy 모델과의 호환성을 위함입니다.

from pydantic import BaseModel
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

    class Config:
        orm_mode = True

class FireEvent(BaseModel):
    id: int
    video_id: str
    timestamp: datetime
    event_type: str
    confidence: Optional[float] = None
    analysis: Optional[str] = None

    class Config:
        orm_mode = True

class SuspectEvent(BaseModel):
    id: int
    video_id: str
    timestamp: datetime
    confidence: float
    analysis: str

    class Config:
        orm_mode = True 