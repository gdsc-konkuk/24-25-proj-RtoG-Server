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

@router.post("/upload", response_model=Video)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Video:
    if not file.content_type in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if file.size > settings.MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        saved_file_path = video_processing_service.save_video(file)
        
        video_id = str(uuid.uuid4())
        db_video = VideoModel(
            id=video_id,
            filename=file.filename,
            location=saved_file_path,
            status="Uploaded",
            created_at=datetime.utcnow(),
            cctv_name="Unknown",
            installation_date="N/A",
            resolution="N/A",
            angle="N/A",
            description=f"Uploaded video: {file.filename}"
        )
        db.add(db_video)
        db.commit()
        db.refresh(db_video)
        
        return db_video

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload video: {e}")

@router.get("/process/{video_id}")
async def process_video(
    video_id: str, 
    db: Session = Depends(get_db)
):
    video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not video.location or not os.path.exists(video.location):
        raise HTTPException(status_code=404, detail=f"Video file not found at location: {video.location}")

    try:
        coordinates = video_processing_service.process_video_extract_coordinates(video.location)
        return {"video_id": video_id, "file_path": video.location, "coordinates": coordinates}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not process video: {e}")

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