from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from services import LiveService
from schemas import LiveResponse
import cv2
import asyncio
import numpy as np
from fastapi.responses import StreamingResponse
# yolo.py 에서 실제 함수 import
from yolo import process_frame_with_yolo
# 설정값 import
from config import settings

router = APIRouter()

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


@router.get("/{video_name}")
async def stream_video_realtime(video_name: str):
    """
    특정 영상을 실시간으로 읽어 YOLO 검증을 수행하고, 마킹된 프레임을 스트리밍합니다.
    """
    # 설정 파일에서 경로 사용
    video_path = f"{settings.VIDEO_STORAGE_PATH}/{video_name}"
    print(f"Attempting to stream video from: {video_path}")
    return StreamingResponse(frame_generator_with_yolo(video_path), media_type="multipart/x-mixed-replace; boundary=frame")

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