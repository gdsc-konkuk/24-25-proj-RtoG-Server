# api-server/routers/videos.py
# 이 파일은 비디오 업로드, 처리, 조회 등 비디오 관련 HTTP API 엔드포인트를 정의합니다.
# FastAPI의 APIRouter를 사용하여 관련 엔드포인트들을 그룹화하고,
# 비디오 데이터 처리를 위해 SQLAlchemy 세션, Pydantic 스키마, 서비스 모듈과 상호작용합니다.
# 주요 기능:
# - 비디오 파일 업로드 (POST /upload)
# - 특정 비디오 처리 요청 (GET /process/{video_id})
# - 전체 비디오 목록 조회 (GET /)
# - 특정 비디오 정보 조회 (GET /{video_id})
# - 특정 비디오 관련 화재 이벤트 조회 (GET /{video_id}/events)

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Tuple, Any # Any 추가
from sqlalchemy.orm import Session
import uuid # uuid 추가
import os # os 추가
from datetime import datetime # datetime 추가

from .. import schemas, models # 스키마 및 모델 경로 변경
from ..services import video_processing_service # 서비스 경로 변경
from ..config import settings # 설정 경로 변경
from ..database import get_db # DB 의존성 경로 변경

router = APIRouter()

@router.post("/upload", summary="Upload a new video", response_model=schemas.Video)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.content_type in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if file.size > settings.MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        # 파일 저장
        # 실제 저장 경로는 VideoProcessingService 내부에서 settings.VIDEO_UPLOAD_DIR 사용
        # file.filename 을 안전하게 처리하는 로직 추가 고려 (예: werkzeug.utils.secure_filename)
        # 여기서는 단순화를 위해 직접 사용
        saved_file_path = video_processing_service.save_video(file)
        
        # DB에 비디오 정보 저장
        video_id = str(uuid.uuid4())
        db_video = models.Video(
            id=video_id,
            filename=file.filename, # 원본 파일명 저장
            location=saved_file_path, # 실제 저장된 전체 경로 또는 상대 경로
            status="Uploaded",
            created_at=datetime.utcnow(), # UTC 시간으로 저장
            # 초기에는 다른 메타데이터는 비워둘 수 있음
            cctv_name="Unknown",
            installation_date="N/A",
            resolution="N/A",
            angle="N/A",
            description=f"Uploaded video: {file.filename}"
        )
        db.add(db_video)
        db.commit()
        db.refresh(db_video)
        
        return db_video # 스키마에 맞는 Video 모델 객체 반환

    except Exception as e:
        # print(f"Error uploading video: {e}") # 로깅 강화
        raise HTTPException(status_code=500, detail=f"Could not upload video: {e}")

@router.get("/process/{video_id}", summary="Process a video to extract coordinates") # 기존 path parameter 방식 유지
async def process_video(
    video_id: str, 
    db: Session = Depends(get_db)
): # 반환 타입은 일단 Dict로 유지
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not video.location or not os.path.exists(video.location):
        raise HTTPException(status_code=404, detail=f"Video file not found at location: {video.location}")

    try:
        # video.location (저장된 실제 파일 경로)을 사용
        coordinates = video_processing_service.process_video_extract_coordinates(video.location)
        # 처리 결과 DB 업데이트 (예: status 변경 또는 결과 저장)
        # video.status = "Processed"
        # db.commit()
        return {"video_id": video_id, "file_path": video.location, "coordinates": coordinates}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # print(f"Error processing video {video_id}: {e}") # 로깅 강화
        raise HTTPException(status_code=500, detail=f"Could not process video: {e}")

@router.get("/", response_model=List[schemas.Video], summary="Get all videos")
async def get_videos(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    videos = db.query(models.Video).offset(skip).limit(limit).all()
    return videos

@router.get("/{video_id}", response_model=schemas.Video, summary="Get a specific video by ID")
async def get_video(
    video_id: str,
    db: Session = Depends(get_db)
):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@router.get("/{video_id}/events", response_model=List[schemas.FireEvent], summary="Get fire events for a video") # 스키마 이름 확인 필요
async def get_fire_events_for_video(
    video_id: str,
    db: Session = Depends(get_db)
):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    # events = db.query(models.FireEvent).filter(models.FireEvent.video_id == video_id).all() # 이렇게 직접 조회하거나
    return video.events # Video 모델에 정의된 relationship 사용

# SuspectEvent 관련 라우트는 FireEvent와 통합되거나 별도 관리될 수 있음.
# 현재 schemas.py 에는 SuspectEvent가 있고, models.py에는 FireEvent, SuspectEvent 둘 다 있음.
# 여기서는 FireEvent를 중심으로 하고, SuspectEvent는 필요시 추가. 