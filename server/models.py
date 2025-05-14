# server/models.py
# SQLAlchemy를 사용한 데이터베이스 모델 정의
# - Video: 비디오 정보
# - FireEvent: 화재 감지 이벤트
# - SuspectEvent: 용의자/특이사항 이벤트

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    location = Column(String)
    status = Column(String, default="Uploaded")
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    cctv_name = Column(String, nullable=True)
    installation_date = Column(String, nullable=True)
    resolution = Column(String, nullable=True)
    angle = Column(String, nullable=True)

    events = relationship("FireEvent", back_populates="video")
    suspect_events = relationship("SuspectEvent", back_populates="video")

class FireEvent(Base):
    __tablename__ = "fire_events"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String)
    confidence = Column(Float, nullable=True)
    analysis = Column(String, nullable=True)
    file_path = Column(String, nullable=True)  # 저장된 영상 파일 경로
    thumbnail_path = Column(String, nullable=True)  # 썸네일 이미지 경로

    video = relationship("Video", back_populates="events")

class SuspectEvent(Base):
    __tablename__ = "suspect_events"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)
    analysis = Column(String)

    video = relationship("Video", back_populates="suspect_events") 