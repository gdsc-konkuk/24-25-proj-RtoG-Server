from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from sqlalchemy.orm import Session
from typing import List, Dict, Set, Optional
import json
import asyncio
from datetime import datetime
import os
import cv2
import numpy as np
from pathlib import Path
import sys
import base64
from ultralytics import YOLO
import collections
from glob import glob

# gemini.py의 경로를 시스템 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from gemini import use_gemini

from . import models, schemas
from .database import engine, get_db, Base

# YOLO 모델 로드
model = YOLO('training/yolo11n.pt')
model.conf = 0.1  # 신뢰도 임계값 설정 (10%)

# 데이터베이스 테이블 생성
print("데이터베이스 테이블 생성 중...")
Base.metadata.drop_all(bind=engine)  # 기존 테이블 삭제
Base.metadata.create_all(bind=engine)  # 새 테이블 생성
print("데이터베이스 테이블 생성 완료")

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/static/templates")

class VideoStream:
    def __init__(self, video_id: str, video_path: str):
        self.video_id = video_id
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        self.is_running = True
        self.fps = 5  # stream_video와 동일하게 5FPS 기준
        self.buffer_size = self.fps * 5  # 5초치 프레임
        self.frame_buffer = collections.deque(maxlen=self.buffer_size)  # 최근 5초 프레임 버퍼
        print(f"비디오 스트림 초기화: {video_id}, 경로: {video_path}")

    async def get_frame(self) -> tuple[bool, str]:
        if not self.is_running:
            return False, ''
        
        success, frame = self.cap.read()
        if not success:
            print(f"비디오 프레임 읽기 실패: {self.video_id}")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오가 끝나면 처음으로
            success, frame = self.cap.read()
            if not success:
                print(f"비디오 재시작 실패: {self.video_id}")
                self.is_running = False
                return False, ''
        
        # 프레임 크기 조정 (선택사항)
        frame = cv2.resize(frame, (640, 480))
        # 버퍼에 프레임 저장 (원본)
        self.frame_buffer.append(frame.copy())
        
        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            print(f"프레임 인코딩 실패: {self.video_id}")
            return False, ''
        
        # base64로 인코딩
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return True, frame_base64

    def get_buffer_frames(self):
        """최근 5초 프레임 반환"""
        return list(self.frame_buffer)

    def release(self):
        self.is_running = False
        self.cap.release()
        print(f"비디오 스트림 종료: {self.video_id}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.video_streams: Dict[str, VideoStream] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.streaming_clients: Dict[str, Set[WebSocket]] = {}
        self.analysis_tasks: Dict[str, asyncio.Task] = {}
        self.suspect_clients: Set[WebSocket] = set()
        self.last_suspect_save_time: Dict[str, float] = {}  # 비디오별 마지막 저장 시각(unixtime)
        self.last_alert_time: Dict[str, float] = {}         # 비디오별 마지막 알림 시각(unixtime)
        self.last_gemini_time: Dict[str, float] = {}        # 비디오별 마지막 Gemini 분석 시각(unixtime)

    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            print(f"WebSocket 연결됨. 현재 연결 수: {len(self.active_connections)}")
        except Exception as e:
            print(f"WebSocket 연결 실패: {str(e)}")

    def disconnect(self, websocket: WebSocket):
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            if websocket in self.suspect_clients:
                self.suspect_clients.remove(websocket)
            for clients in self.streaming_clients.values():
                if websocket in clients:
                    clients.remove(websocket)
            print(f"WebSocket 연결 해제됨. 현재 연결 수: {len(self.active_connections)}")
        except Exception as e:
            print(f"WebSocket 연결 해제 실패: {str(e)}")

    async def subscribe_to_suspects(self, websocket: WebSocket):
        try:
            self.suspect_clients.add(websocket)
            print(f"화재 감지 구독 추가됨. 현재 구독자 수: {len(self.suspect_clients)}")
        except Exception as e:
            print(f"화재 감지 구독 실패: {str(e)}")

    async def unsubscribe_from_suspects(self, websocket: WebSocket):
        try:
            self.suspect_clients.discard(websocket)
            print(f"화재 감지 구독 해제됨. 현재 구독자 수: {len(self.suspect_clients)}")
        except Exception as e:
            print(f"화재 감지 구독 해제 실패: {str(e)}")

    async def notify_suspects(self, video_id: str, timestamp: str, confidence: float, analysis: str, frame: str):
        disconnected_clients = set()
        for client in self.suspect_clients:
            try:
                await client.send_json({
                    "event": "fire_detected",
                    "data": {
                        "id": video_id,
                        "timestamp": timestamp,
                        "confidence": confidence,
                        "analysis": analysis,
                        "frame": frame
                    }
                })
            except Exception as e:
                print(f"알림 전송 실패: {str(e)}")
                disconnected_clients.add(client)
        
        # 연결이 끊긴 클라이언트 제거
        for client in disconnected_clients:
            self.suspect_clients.discard(client)

    async def start_streaming(self, video_id: str, video_path: str):
        """비디오 스트리밍 시작"""
        try:
            if video_id in self.video_streams:
                print(f"이미 실행 중인 스트림: {video_id}")
                return
            
            print(f"새로운 비디오 스트림 시작: {video_id}")
            stream = VideoStream(video_id, video_path)
            self.video_streams[video_id] = stream
            self.streaming_clients[video_id] = set()
            
            # 스트리밍 태스크 시작
            task = asyncio.create_task(self.stream_video(video_id))
            self.streaming_tasks[video_id] = task
            print(f"스트리밍 태스크 생성됨: {video_id}")
            
            # 분석 태스크 시작
            analysis_task = asyncio.create_task(self.analyze_video(video_id, video_path))
            self.analysis_tasks[video_id] = analysis_task
            print(f"분석 태스크 생성됨: {video_id}")
        except Exception as e:
            print(f"스트리밍 시작 중 오류 발생: {video_id}, 오류: {str(e)}")
            if video_id in self.video_streams:
                self.video_streams[video_id].release()
                del self.video_streams[video_id]
            raise

    async def analyze_video(self, video_id: str, video_path: str):
        """비디오 분석 및 화재 감지"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        db = next(get_db())  # DB 세션 생성
        
        # VideoStream 인스턴스 가져오기
        stream = self.video_streams.get(video_id)
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                if frame_count % 25 == 0:  # 5FPS 기준 5초마다 검사
                    is_fire, confidence = detect_fire(frame)
                    
                    if is_fire and confidence >= 0.1:
                        import time
                        now = time.time()
                        last_alert = self.last_alert_time.get(video_id, 0)
                        # 10초 쿨타임이 남아있으면 Gemini 분석/알림 모두 생략
                        if now - last_alert < 10:
                            print(f"[{video_id}] 10초 쿨타임 중: Gemini/알림 생략")
                        else:
                            try:
                                temp_frame_path = f"temp_frame_{video_id}.jpg"
                                cv2.imwrite(temp_frame_path, frame)
                                risk_level = use_gemini(temp_frame_path)
                                os.remove(temp_frame_path)
                                print(f"[{video_id}] Gemini 분석 결과: {risk_level}")
                                if risk_level == '위험' or risk_level == '주의':
                                    # 화재로 판단된 경우에만 10초 쿨타임 시작 및 프론트 알림
                                    self.last_alert_time[video_id] = now
                                    frame_base64 = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
                                    await self.notify_suspects(
                                        video_id=video_id,
                                        timestamp=datetime.now().isoformat(),
                                        confidence=confidence,
                                        analysis=risk_level,
                                        frame=frame_base64
                                    )
                                    # 의심 영상 저장 (30초 제한)
                                    last_save = self.last_suspect_save_time.get(video_id, 0)
                                    if now - last_save >= 30:
                                        self.last_suspect_save_time[video_id] = now
                                        try:
                                            suspects_dir = Path("app/static/suspects")
                                            suspects_dir.mkdir(parents=True, exist_ok=True)
                                            suspect_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                                            suspect_video_path = str(suspects_dir / f"suspect_{video_id}_{suspect_time}.mp4")
                                            suspect_thumb_path = str(suspects_dir / f"suspect_{video_id}_{suspect_time}.jpg")
                                            prev_frames = stream.get_buffer_frames() if stream else []
                                            next_frames = []
                                            for _ in range(25):
                                                ret, next_frame = cap.read()
                                                if not ret:
                                                    break
                                                next_frame = cv2.resize(next_frame, (640, 480))
                                                next_frames.append(next_frame)
                                            all_frames = prev_frames + [frame] + next_frames
                                            if all_frames:
                                                cv2.imwrite(suspect_thumb_path, all_frames[0])
                                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                                out = cv2.VideoWriter(suspect_video_path, fourcc, 5, (640, 480))
                                                for f in all_frames:
                                                    out.write(f)
                                                out.release()
                                                print(f"의심 영상 저장: {suspect_video_path}")
                                                print(f"썸네일 저장: {suspect_thumb_path}")
                                        except Exception as e:
                                            print(f"Error in suspect video save: {str(e)}")
                                    else:
                                        print(f"[{video_id}] 30초 내 중복 저장 방지: 저장하지 않음")
                            except Exception as e:
                                print(f"Gemini 분석 오류: {str(e)}")
                        # 1분 제한 영상 저장은 기존대로
                        last_save = self.last_suspect_save_time.get(video_id, 0)
                        if now - last_save >= 60:
                            self.last_suspect_save_time[video_id] = now
                            try:
                                suspects_dir = Path("app/static/suspects")
                                suspects_dir.mkdir(parents=True, exist_ok=True)
                                suspect_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                                suspect_video_path = str(suspects_dir / f"suspect_{video_id}_{suspect_time}.mp4")
                                suspect_thumb_path = str(suspects_dir / f"suspect_{video_id}_{suspect_time}.jpg")
                                prev_frames = stream.get_buffer_frames() if stream else []
                                next_frames = []
                                for _ in range(25):
                                    ret, next_frame = cap.read()
                                    if not ret:
                                        break
                                    next_frame = cv2.resize(next_frame, (640, 480))
                                    next_frames.append(next_frame)
                                all_frames = prev_frames + [frame] + next_frames
                                if all_frames:
                                    cv2.imwrite(suspect_thumb_path, all_frames[0])
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    out = cv2.VideoWriter(suspect_video_path, fourcc, 5, (640, 480))
                                    for f in all_frames:
                                        out.write(f)
                                    out.release()
                                    print(f"의심 영상 저장: {suspect_video_path}")
                                    print(f"썸네일 저장: {suspect_thumb_path}")
                            except Exception as e:
                                print(f"Error in suspect video save: {str(e)}")
                        else:
                            print(f"[{video_id}] 1분 내 중복 저장 방지: 저장하지 않음")
                
                frame_count += 1
                await asyncio.sleep(1/30)  # 30 FPS 유지
                
        finally:
            cap.release()
            db.close()  # DB 세션 종료

    async def stream_video(self, video_id: str):
        """비디오 스트리밍 처리"""
        try:
            stream = self.video_streams.get(video_id)
            if not stream:
                print(f"스트림을 찾을 수 없음: {video_id}")
                return

            while stream.is_running:
                if not self.streaming_clients.get(video_id):
                    print(f"구독자 없음: {video_id}")
                    await asyncio.sleep(1)
                    continue

                success, frame_base64 = await stream.get_frame()
                if not success:
                    print(f"프레임 가져오기 실패: {video_id}")
                    await asyncio.sleep(0.1)
                    continue

                # 모든 구독자에게 프레임 전송
                disconnected_clients = set()
                for client in self.streaming_clients[video_id]:
                    try:
                        await client.send_json({
                            "type": "frame",
                            "video_id": video_id,
                            "frame": frame_base64
                        })
                    except Exception as e:
                        print(f"프레임 전송 실패: {str(e)}")
                        disconnected_clients.add(client)

                # 연결이 끊긴 클라이언트 제거
                for client in disconnected_clients:
                    self.streaming_clients[video_id].discard(client)

                await asyncio.sleep(0.1)  # 10 FPS로 조정 (속도 개선)

        except Exception as e:
            print(f"스트리밍 중 오류 발생: {video_id}, 오류: {str(e)}")
        finally:
            if video_id in self.video_streams:
                self.video_streams[video_id].release()
                del self.video_streams[video_id]
            if video_id in self.streaming_tasks:
                del self.streaming_tasks[video_id]
            if video_id in self.streaming_clients:
                del self.streaming_clients[video_id]

    async def subscribe_to_video(self, websocket: WebSocket, video_id: str):
        try:
            if video_id not in self.streaming_clients:
                self.streaming_clients[video_id] = set()
            self.streaming_clients[video_id].add(websocket)
            print(f"비디오 구독 추가: {video_id}, 현재 구독자 수: {len(self.streaming_clients[video_id])}")
        except Exception as e:
            print(f"비디오 구독 실패: {str(e)}")

manager = ConnectionManager()

def detect_fire(frame: np.ndarray) -> tuple[bool, float]:
    """화재 감지 함수 (YOLOv11 모델 사용)"""
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            max_confidence = float(boxes.conf.max())
            return True, max_confidence
    
    return False, 0.0

def initialize_database():
    """서버 시작 시 데이터베이스 초기화"""
    print("데이터베이스 초기화 시작...")
    db = next(get_db())
    try:
        # 메타데이터 파일 읽기
        metadata_path = Path("app/static/metadata/videos.json")
        if not metadata_path.exists():
            print("Warning: Metadata file not found")
            return
        
        print(f"메타데이터 파일 읽기: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"메타데이터에서 {len(metadata['videos'])}개의 비디오 정보를 찾았습니다.")
        
        # 각 비디오 메타데이터를 DB에 저장
        for video_metadata in metadata["videos"]:
            video_id = video_metadata["id"]
            print(f"비디오 처리 중: {video_id}")
            
            # 기존 비디오 정보 조회
            existing_video = db.query(models.Video).filter(models.Video.id == video_id).first()
            
            if existing_video:
                # 기존 정보 업데이트
                print(f"기존 비디오 정보 업데이트: {video_id}")
                for key, value in video_metadata.items():
                    setattr(existing_video, key, value)
            else:
                # 새 비디오 정보 생성
                print(f"새 비디오 정보 생성: {video_id}")
                db_video = models.Video(**video_metadata)
                db.add(db_video)
        
        db.commit()
        print("데이터베이스 초기화 완료")
        
    except Exception as e:
        print(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
        db.rollback()
    finally:
        db.close()

# 서버 시작 시 DB 초기화
initialize_database()

@app.get("/videos", response_model=List[schemas.Video])
async def get_videos(db: Session = Depends(get_db)):
    """비디오 목록 조회"""
    try:
        videos = db.query(models.Video).all()
        print(f"데이터베이스에서 {len(videos)}개의 비디오를 찾았습니다.")
        
        # 각 비디오에 대해 스트리밍 시작
        for video in videos:
            video_path = f"app/static/videos/{video.filename}"
            print(f"비디오 경로 확인: {video_path}")
            
            if os.path.exists(video_path):
                print(f"비디오 파일 존재: {video_path}")
                try:
                    await manager.start_streaming(video.id, video_path)
                    print(f"비디오 스트리밍 시작됨: {video.id}")
                except Exception as e:
                    print(f"비디오 스트리밍 시작 실패: {video.id}, 오류: {str(e)}")
            else:
                print(f"비디오 파일을 찾을 수 없음: {video_path}")
        
        return videos
    except Exception as e:
        print(f"비디오 목록 조회 중 오류 발생: {str(e)}")
        raise

@app.websocket("/ws/suspects")
async def suspects_endpoint(websocket: WebSocket):
    try:
        await manager.connect(websocket)
        await manager.subscribe_to_suspects(websocket)
        print("화재 감지 웹소켓 연결됨")
        
        while True:
            try:
                data = await websocket.receive_json()
                if data["action"] == "unsubscribe":
                    await manager.unsubscribe_from_suspects(websocket)
                    break
            except Exception as e:
                print(f"메시지 수신 중 오류: {str(e)}")
                break
    except WebSocketDisconnect:
        print("화재 감지 웹소켓 연결 종료")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"화재 감지 웹소켓 오류: {str(e)}")
        manager.disconnect(websocket)

@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    try:
        await manager.connect(websocket)
        print("스트림 웹소켓 연결됨")
        
        while True:
            try:
                data = await websocket.receive_json()
                if data["action"] == "subscribe":
                    video_id = data["video_id"]
                    await manager.subscribe_to_video(websocket, video_id)
                elif data["action"] == "unsubscribe":
                    video_id = data["video_id"]
                    if video_id in manager.streaming_clients:
                        manager.streaming_clients[video_id].discard(websocket)
                    break
            except Exception as e:
                print(f"메시지 수신 중 오류: {str(e)}")
                break
    except WebSocketDisconnect:
        print("스트림 웹소켓 연결 종료")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"스트림 웹소켓 오류: {str(e)}")
        manager.disconnect(websocket)

@app.get("/suspects/{id}/events")
async def get_suspect_events(id: str, db: Session = Depends(get_db)):
    events = db.query(models.FireEvent).filter(models.FireEvent.video_id == id).all()
    if not events:
        raise HTTPException(status_code=404, detail="Events not found")
    
    return {
        "id": id,
        "events": [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "confidence": event.confidence,
                "analysis": event.analysis
            }
            for event in events
        ]
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/records")
async def get_records():
    """의심 영상(10초) 및 썸네일 목록 반환"""
    suspects_dir = Path("app/static/suspects")
    records = []
    for video_path in sorted(suspects_dir.glob("*.mp4")):
        base = video_path.stem  # suspect_video_001_20250510_173126
        thumb_path = suspects_dir / f"{base}.jpg"
        # 파일명에서 정보 추출
        parts = base.split("_")
        video_id = parts[2] if len(parts) > 2 else ""
        date_str = "_".join(parts[3:]) if len(parts) > 3 else ""
        records.append({
            "video_id": video_id,
            "datetime": date_str,
            "video_url": f"/static/suspects/{base}.mp4",
            "thumb_url": f"/static/suspects/{base}.jpg"
        })
    return JSONResponse(records)

@app.get("/suspect_video/{filename}")
async def get_suspect_video(filename: str):
    file_path = f"app/static/suspects/{filename}"
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(file_path, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 