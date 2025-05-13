# server/services.py
# 이 파일은 애플리케이션의 주요 비즈니스 로직을 서비스 계층의 클래스들로 분리하여 정의합니다.
# 각 서비스 클래스는 특정 도메인 관련 기능을 캡슐화하여 재사용성과 테스트 용이성을 높입니다.
# 주요 서비스:
# - VideoProcessingService: 비디오 파일 저장 및 처리를 담당합니다.
# - LiveService: 실시간 CCTV 스트리밍 목록을 제공합니다.
# - RecordService: 화재 이벤트 기록을 관리합니다.

import os
from typing import Any
import cv2
import asyncio
import numpy as np
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from yolo import process_frame_with_yolo

from ultralytics import YOLO
from datetime import datetime, timedelta

from config import settings 
from models import Video, FireEvent


class VideoProcessingService:
    """비디오 파일 처리 및 프레임 데이터 관리를 위한 서비스"""
    def __init__(self):
        self.upload_dir = settings.VIDEO_UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_video(self, file: Any) -> str:
        """업로드된 비디오 파일을 저장하고 경로를 반환합니다."""
        file_path = os.path.join(self.upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)
        return file_path

    @staticmethod
    async def frame_generator_with_yolo(video_path: str):
        """
        주어진 영상 파일 경로를 통해 프레임을 읽고, YOLO 검증 후 마킹된 프레임을 생성하여 yield 합니다.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file at {video_path}")
                raise HTTPException(status_code=404, detail="영상을 열 수 없습니다.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Info: Reached end of video or failed to read frame.")
                    break  # 영상의 끝 또는 프레임 읽기 실패

                # YOLO 검증 및 프레임 마킹
                try:
                    marked_frame = await process_frame_with_yolo(frame)
                except Exception as yolo_err:
                    print(f"Error during YOLO processing: {yolo_err}")
                    # 오류 발생 시 원본 프레임을 전송합니다.
                    marked_frame = frame

                if marked_frame is None or marked_frame.size == 0:
                    print("Warning: Marked frame is empty.")
                    continue

                # 마킹된 프레임을 JPEG 형식으로 인코딩
                ret, encoded_frame = cv2.imencode('.jpg', marked_frame)
                if not ret:
                    print("Warning: Failed to encode frame to JPEG.")
                    continue

                # 스트리밍 형식에 맞게 yield
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
                await asyncio.sleep(0.01) # 스트리밍 간격 조절 (필요에 따라 조정)

        except FileNotFoundError:
            print(f"Error: Video file not found at {video_path}")
            raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            print(f"Error in frame generator: {e}")
            raise HTTPException(status_code=500, detail=f"프레임 처리 또는 스트리밍 중 오류 발생: {e}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
                print("Info: Video capture released.")

class LiveService:
    """실시간 CCTV 스트리밍 가능 목록 제공"""

    @staticmethod
    def get_lives(db) -> list[dict]:
        """데이터베이스에서 비디오 목록을 가져와 반환합니다."""
        videos = db.query(Video).all()
        return [
            {
                "id": f"cctv_{v.id:03d}",
                "name": v.cctv_name or v.filename,
                "address": v.location or "",
                "video_id": v.id
            }
            for v in videos
        ]

class RecordService:
    """화재 이벤트 기록 관련 서비스"""
    
    @staticmethod
    def get_records(db, start=None, end=None):
        """일자별로 그룹화된 화재 이벤트 목록을 반환합니다."""
        try:
            query = db.query(FireEvent).join(Video)
            
            if start:
                try:
                    start_date = datetime.strptime(start, "%Y-%m-%d")
                    query = query.filter(FireEvent.timestamp >= start_date)
                except ValueError:
                    raise ValueError("Invalid start date format. Use YYYY-MM-DD.")
            if end:
                try:
                    end_date = datetime.strptime(end, "%Y-%m-%d")
                    query = query.filter(FireEvent.timestamp < end_date + timedelta(days=1))
                except ValueError:
                    raise ValueError("Invalid end date format. Use YYYY-MM-DD.")
                
            events = query.order_by(FireEvent.timestamp.desc()).all()
            
            # 일자별로 그룹화
            daily_events = {}
            for event in events:
                date_str = event.timestamp.strftime("%Y-%m-%d")
                if date_str not in daily_events:
                    daily_events[date_str] = []
                    
                daily_events[date_str].append({
                    "eventId": f"evt_{event.id:03d}",
                    "cctv_name": event.video.cctv_name or event.video.filename,
                    "address": event.video.location or "주소 정보 없음",
                    "timestamp": event.timestamp.isoformat()
                })
                
            # 날짜순으로 정렬하여 반환
            return [
                {"date": date, "events": event_list}
                for date, event_list in sorted(daily_events.items(), reverse=True)
            ]
        except ValueError as e:
            raise ValueError(f"Invalid date format: {str(e)}")
        except Exception as e:
            print(f"Error fetching records: {e}")
            raise Exception("Error fetching records")

    @staticmethod
    def get_record_detail(db, event_id):
        """특정 화재 이벤트의 상세 정보를 반환합니다."""
        try:
            if not event_id.startswith("evt_") or not event_id[4:].isdigit():
                raise ValueError("Invalid event ID format. Expected 'evt_XXX'.")
            numeric_id = int(event_id[4:])

            event = db.query(FireEvent).join(Video).filter(FireEvent.id == numeric_id).first()
            if not event:
                return None

            return {
                "eventId": f"evt_{event.id:03d}",
                "cctv_name": event.video.cctv_name or event.video.filename,
                "address": event.video.location or "주소 정보 없음",
                "timestamp": event.timestamp.isoformat(),
                "description": event.analysis or "상세 분석 정보 없음"
            }
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            print(f"Error fetching record detail for {event_id}: {e}")
            raise Exception("Error fetching record detail") 