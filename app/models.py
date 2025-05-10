from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    location = Column(String)
    cctv_name = Column(String)
    installation_date = Column(String)
    resolution = Column(String)
    angle = Column(String)
    status = Column(String)

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