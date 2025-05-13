# server/routers/videos.py
# 비디오 관련 HTTP API 엔드포인트 정의
# - 비디오 업로드, 처리, 조회 등의 기능 제공

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import List
from sqlalchemy.orm import Session
import uuid
import os
from datetime import datetime
import cv2
import asyncio
from fastapi.responses import StreamingResponse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Video, FireEvent, RecordsResponse, EventDetail
from models import Video as VideoModel, FireEvent as FireEventModel
from services import RecordService, VideoProcessingService
from database import get_db
from config import settings

router = APIRouter()

@router.get("", response_model=RecordsResponse)
async def get_records(
    start: str = Query(None, description="조회 시작 날짜(YYYY-MM-DD)"),
    end: str = Query(None, description="조회 종료 날짜(YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    화재 이벤트 기록을 일자별로 그룹화하여 반환합니다.
    
    특징:
    - 일자별로 그룹화된 이벤트 목록 제공
    - 각 이벤트는 썸네일, CCTV 정보, 발생 시각 포함
    - start, end 파라미터로 날짜 범위 필터링 가능
    
    반환 예시:
    ```json
    {
      "records": [
        {
          "date": "2024-03-15",
          "events": [
            {
              "eventId": "evt_001",
              "cctv_name": "강릉시청 앞 CCTV-1",
              "address": "강원도 강릉시",
              "thumbnail_url": "/static/events/evt_001.jpg",
              "timestamp": "2024-03-15T14:23:00"
            }
          ]
        },
        {
          "date": "2024-03-14",
          "events": [
            {
              "eventId": "evt_002",
              "cctv_name": "강릉시청 앞 CCTV-2",
              "address": "강원도 강릉시",
              "thumbnail_url": "/static/events/evt_002.jpg",
              "timestamp": "2024-03-14T15:30:00"
            }
          ]
        }
      ]
    }
    ```
    """
    result = RecordService.get_records(db, start, end)
    
    # thumbnail_url 필드 추가
    for record in result:
        for event in record["events"]:
            if "thumbnail_url" not in event:
                event["thumbnail_url"] = f"/static/record/events/{event['eventId']}_thumb.jpg"
    
    return {"records": result}

@router.get("/{eventId}")
async def get_record_detail(eventId: int, db: Session = Depends(get_db)):
    """특정 이벤트의 저장된 영상을 스트리밍합니다."""
    print(f"\n=== Debug: get_record_detail ===")
    print(f"Requested eventId: {eventId}")
    
    event = db.query(FireEventModel).filter(FireEventModel.id == eventId).first()
    if not event:
        print(f"Event not found in database")
        raise HTTPException(status_code=404, detail="Event not found")
    
    print(f"Found event: {event.id}")
    print(f"Event file_path: {event.file_path}")
    
    if not event.file_path:
        print(f"Event file_path is None")
        raise HTTPException(status_code=404, detail="Event video not found")
    
    # 파일 경로가 이미 static/record/events/event_YYYYMMDD_HHMMSS.mp4 형식으로 저장되어 있음
    video_path = os.path.abspath(event.file_path)
    print(f"Absolute video path: {video_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    if not os.path.exists(video_path):
        print(f"Video file not found at: {video_path}")
        raise HTTPException(status_code=404, detail=f"Video file not found at {video_path}")
    
    print(f"Video file exists, starting stream")
    print("=== End Debug ===\n")
    
    return StreamingResponse(
        VideoProcessingService.video_stream(video_path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# SuspectEvent 관련 라우트는 FireEvent와 통합되거나 별도 관리될 수 있음.
# 현재 schemas.py 에는 SuspectEvent가 있고, models.py에는 FireEvent, SuspectEvent 둘 다 있음.
# 여기서는 FireEvent를 중심으로 하고, SuspectEvent는 필요시 추가. 
