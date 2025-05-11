# server/routers/videos.py
# 비디오 관련 HTTP API 엔드포인트 정의
# - 비디오 업로드, 처리, 조회 등의 기능 제공

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session
import uuid
import os
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Video, FireEvent
from models import Video as VideoModel, FireEvent as FireEventModel
from services import video_processing_service
from config import settings
from database import get_db

router = APIRouter()

@router.get("/", response_model=List[Video])
async def get_videos(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    videos = db.query(VideoModel).offset(skip).limit(limit).all()
    return videos

@router.get("/{video_id}", response_model=Video)
async def get_video(
    video_id: str,
    db: Session = Depends(get_db)
):
    video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@router.get("/{video_id}/events", response_model=List[FireEvent])
async def get_video_events(video_id: str, db: Session = Depends(get_db)):
    video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video.events

# SuspectEvent 관련 라우트는 FireEvent와 통합되거나 별도 관리될 수 있음.
# 현재 schemas.py 에는 SuspectEvent가 있고, models.py에는 FireEvent, SuspectEvent 둘 다 있음.
# 여기서는 FireEvent를 중심으로 하고, SuspectEvent는 필요시 추가. 