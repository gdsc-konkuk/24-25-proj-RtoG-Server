from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from services import LiveService, VideoProcessingService
from schemas import LiveResponse
from fastapi.responses import StreamingResponse
from config import settings
from models import Video

router = APIRouter()

@router.get("/{video_id}")
async def stream_video_realtime(video_id: str, db: Session = Depends(get_db)):
    """
    특정 영상을 실시간으로 읽어 YOLO 검증을 수행하고, 마킹된 프레임을 스트리밍합니다.
    """
    # DB에서 비디오 정보 조회
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # 비디오 파일 경로
    video_path = f"{settings.VIDEO_STORAGE_PATH}/{video.id}.mp4"
    print(f"Attempting to stream video from: {video_path}")
    return StreamingResponse(
        VideoProcessingService.frame_generator_with_yolo(video_path), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("", response_model=LiveResponse)
async def lives_endpoint(db: Session = Depends(get_db)):
    """
    실시간 스트리밍 가능한 CCTV 목록을 반환합니다.
    
    특징:
    - Live 탭 진입 시 프론트엔드에서 호출
    - 각 CCTV의 ID, 이름, 설치 위치(address)를 포함
    
    반환 예시:
    ```json
    {
      "cctvs": [
        {
          "id": "video_001",
          "name": "강릉시청 앞 CCTV-1",
          "address": "강원도 강릉시"
        },
        {
          "id": "video_002",
          "name": "강릉시청 앞 CCTV-2",
          "address": "강원도 강릉시"
        }
      ]
    }
    ```
    """
    return {"cctvs": LiveService.get_lives(db)} 