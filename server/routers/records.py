# server/routers/videos.py
# 비디오 관련 HTTP API 엔드포인트 정의
# - 비디오 업로드, 처리, 조회 등의 기능 제공

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import List
from sqlalchemy.orm import Session
import uuid
import os
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Video, FireEvent, RecordsResponse, EventDetail
from models import Video as VideoModel, FireEvent as FireEventModel
from services import video_processing_service, RecordService
from config import settings
from database import get_db

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
    return {"records": result}

@router.get("/{eventId}", response_model=EventDetail)
async def get_record_detail(eventId: str, db: Session = Depends(get_db)):
    """
    특정 화재 이벤트의 상세 정보를 반환합니다.
    
    특징:
    - 이벤트 영상 URL 제공
    - CCTV 정보 및 설치 위치 포함
    - 발생 시각 및 상세 설명 제공
    
    반환 예시:
    ```json
    {
      "eventId": "evt_001",
      "cctv_name": "강릉시청 앞 CCTV-1",
      "address": "강원도 강릉시",
      "timestamp": "2024-03-15T14:23:00",
      "video_url": "/static/events/evt_001.mp4",
      "description": "화재 감지 이벤트 상세 설명"
    }
    ```
    """
    detail = RecordService.get_record_detail(db, eventId)
    if not detail:
        raise HTTPException(status_code=404, detail="Event not found")
    return detail

# SuspectEvent 관련 라우트는 FireEvent와 통합되거나 별도 관리될 수 있음.
# 현재 schemas.py 에는 SuspectEvent가 있고, models.py에는 FireEvent, SuspectEvent 둘 다 있음.
# 여기서는 FireEvent를 중심으로 하고, SuspectEvent는 필요시 추가. 
