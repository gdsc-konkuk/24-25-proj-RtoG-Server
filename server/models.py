# server/models.py
# 이 파일은 SQLAlchemy 모델을 정의하여 데이터베이스의 테이블 구조를 나타냅니다.
# database.py에서 정의된 Base 클래스를 상속받아 각 테이블(엔티티)에 대한 클래스를 생성합니다.
# 주요 모델:
# - Video: 업로드된 비디오의 메타데이터 (ID, 파일명, 저장 위치, 상태 등)를 저장합니다.
# - FireEvent: 감지된 화재 이벤트 정보 (ID, 비디오 ID, 타임스탬프, 분석 결과 등)를 저장합니다.
# - SuspectEvent: 감지된 용의자/특이사항 이벤트 정보 (ID, 비디오 ID, 타임스탬프 등)를 저장합니다.
# 모델 간의 관계(relationship)도 정의하여 ORM 기능을 활용할 수 있도록 합니다.

from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    location = Column(String)
    cctv_name = Column(String)
    installation_date = Column(String)
    resolution = Column(String)
    angle = Column(String)
    status = Column(String)
    description = Column(String)
    created_at = Column(DateTime)

    events = relationship("FireEvent", back_populates="video")

class FireEvent(Base):
    __tablename__ = "fire_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.id"))
    timestamp = Column(DateTime, default=datetime.now)
    event_type = Column(String)
    confidence = Column(Float)
    analysis = Column(String)

    video = relationship("Video", back_populates="events")

class SuspectEvent(Base):
    __tablename__ = "suspect_events"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    video_id = Column(String)
    timestamp = Column(DateTime)
    confidence = Column(Float)
    analysis = Column(String) 