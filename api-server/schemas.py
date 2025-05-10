from pydantic import BaseModel
from typing import Optional

class Video(BaseModel):
    id: str
    filename: str
    description: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        orm_mode = True

class SuspectEvent(BaseModel):
    id: int
    video_id: str
    timestamp: str
    confidence: float
    analysis: str

    class Config:
        orm_mode = True 