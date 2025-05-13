from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from services import LiveService, VideoProcessingService
from schemas import LiveResponse
from fastapi.responses import StreamingResponse
from config import settings

router = APIRouter()

@router.get("/{video_name}")
async def stream_video_realtime(video_name: str):
    """
    특정 영상을 실시간으로 읽어 YOLO 검증을 수행하고, 마킹된 프레임을 스트리밍합니다.
    """
    # 설정 파일에서 경로 사용
    video_path = f"{settings.VIDEO_STORAGE_PATH}/{video_name}"
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
    - 각 CCTV의 ID, 이름, 설치 위치(address), WebSocket ID를 포함
    
    반환 예시:
    ```json
    {
      "cctvs": [
        {
          "id": "cctv_001",
          "name": "강릉시청 앞 CCTV-1",
          "address": "강원도 강릉시",
          "socket_id": "ws_001"
        },
        {
          "id": "cctv_002",
          "name": "강릉시청 앞 CCTV-2",
          "address": "강원도 강릉시",
          "socket_id": "ws_002"
        }
      ]
    }
    ```
    """
    return {"cctvs": LiveService.get_lives(db)} 