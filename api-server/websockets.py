# api-server/websockets.py
# 이 파일은 WebSocket 연결 관리 및 실시간 메시지 처리를 위한 UnifiedConnectionManager 클래스를 정의합니다.
# 이 매니저는 다음과 같은 기능을 수행합니다:
# - 클라이언트 WebSocket 연결 수락 및 관리 (connect, disconnect)
# - 특정 이벤트(예: 용의자 발생)에 대한 클라이언트 구독/구독 해제 (subscribe_to_suspects, unsubscribe_from_suspects)
# - 구독한 클라이언트들에게 실시간 이벤트 메시지 브로드캐스팅 (broadcast_suspect_event)
# - 특정 비디오에 대한 스트리밍 세션 시작 및 관리 (start_streaming_session, _stream_video_frames)
# - 비디오 스트리밍 중 실시간 프레임 분석 (YOLO, Gemini) 및 관련 이벤트 처리 (_analyze_video_feed)
# - 클라이언트의 비디오 스트림 구독/구독 해제 (subscribe_to_video_stream, unsubscribe_from_video_stream)
# - 스트리밍 세션 종료 및 관련 자원 정리 (stop_streaming_session)
# services.py의 StreamingService, AnalysisService, VideoProcessingService와 상호작용하여
# 실시간 비디오 처리 및 알림 기능을 구현합니다.

from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Set, Optional, Any
import asyncio
import json
import time
from sqlalchemy.orm import Session
from . import models, schemas
from .database import get_db # get_db는 Depends 용도이므로 직접 호출하지 않음
from .services import StreamingService, video_processing_service, analysis_service
from .config import settings # settings import 추가
from datetime import datetime
import os
import cv2
import base64

class UnifiedConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.video_streams: Dict[str, StreamingService] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.streaming_clients: Dict[str, Set[WebSocket]] = {}
        self.suspect_clients: Set[WebSocket] = set()
        self.analysis_tasks: Dict[str, asyncio.Task] = {}
        self.last_suspect_save_time: Dict[str, float] = {}
        self.last_alert_time: Dict[str, float] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected: {websocket.client}. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if websocket in self.suspect_clients:
            self.suspect_clients.remove(websocket)
            
        # 스트리밍 클라이언트 목록에서도 제거
        # disconnect 호출 시 어떤 video_id에 대한 구독 해제인지 알 수 없으므로, 모든 스트림에서 제거 시도
        for video_id in list(self.streaming_clients.keys()): 
            if websocket in self.streaming_clients.get(video_id, set()):
                self.streaming_clients[video_id].remove(websocket)
                if not self.streaming_clients[video_id]:
                    print(f"No more clients for video {video_id}. Consider stopping stream if auto-stop is implemented.")
                    # del self.streaming_clients[video_id] # 필요시, 또는 stop_streaming_session에서 처리
        
        print(f"WebSocket disconnected: {websocket.client}. Total: {len(self.active_connections)}")

    async def subscribe_to_suspects(self, websocket: WebSocket):
        self.suspect_clients.add(websocket)
        print(f"Suspects subscription added: {websocket.client}. Total subscribers: {len(self.suspect_clients)}")

    async def unsubscribe_from_suspects(self, websocket: WebSocket):
        self.suspect_clients.discard(websocket)
        print(f"Suspects subscription removed: {websocket.client}. Total subscribers: {len(self.suspect_clients)}")

    async def broadcast_suspect_event(self, video_id: str, timestamp: str, confidence: float, analysis: str, frame_base64: str):
        message = {
            "event": "fire_detected",
            "data": {
                "id": video_id,
                "timestamp": timestamp,
                "confidence": confidence,
                "analysis": analysis,
                "frame": frame_base64
            }
        }
        disconnected_clients = set()
        for client in list(self.suspect_clients): # 순회 중 변경 대비 list로 복사
            try:
                await client.send_json(message)
            except WebSocketDisconnect:
                disconnected_clients.add(client)
            except Exception as e:
                print(f"Error broadcasting suspect event to {client.client}: {e}")
                disconnected_clients.add(client)
        
        for client in disconnected_clients:
            self.disconnect(client)
            print(f"Disconnected client {client.client} due to error during suspect broadcast.")

    async def start_streaming_session(self, video_id: str, video_path: str, db: Session):
        if video_id in self.video_streams and self.video_streams[video_id].is_running:
            print(f"Stream already running for {video_id}")
            return

        print(f"Attempting to start new video stream for {video_id} at {video_path}")
        stream_service = StreamingService(video_id, video_path)
        if not stream_service.is_running:
            print(f"Failed to initialize StreamingService for {video_id} at {video_path}")
            video_obj = db.query(models.Video).filter(models.Video.id == video_id).first()
            if video_obj:
                video_obj.status = "Error: File not found or cannot be opened"
                db.commit()
            return

        self.video_streams[video_id] = stream_service
        if video_id not in self.streaming_clients:
            self.streaming_clients[video_id] = set()

        if video_id not in self.streaming_tasks or self.streaming_tasks[video_id].done():
            self.streaming_tasks[video_id] = asyncio.create_task(self._stream_video_frames(video_id))
            print(f"Video streaming task created/restarted for {video_id}")

        if video_id not in self.analysis_tasks or self.analysis_tasks[video_id].done():
            self.analysis_tasks[video_id] = asyncio.create_task(self._analyze_video_feed(video_id, db))
            print(f"Video analysis task created/restarted for {video_id}")
        
        video_obj = db.query(models.Video).filter(models.Video.id == video_id).first()
        if video_obj:
            video_obj.status = "Streaming"
            db.commit()
        else:
            print(f"Warning: Video with id {video_id} not found in DB when starting stream.")

    async def _stream_video_frames(self, video_id: str):
        stream = self.video_streams.get(video_id)
        if not stream or not stream.is_running:
            print(f"Stream for {video_id} not found or not running in _stream_video_frames.")
            return

        print(f"Starting to stream frames for {video_id}")
        try:
            while stream.is_running:
                if not self.streaming_clients.get(video_id):
                    await asyncio.sleep(1)
                    if not self.streaming_clients.get(video_id): # 클라이언트가 없으면 프레임 전송 일시 중지
                        continue # 루프는 계속 돌면서 클라이언트 연결을 기다림
                
                success, frame_base64 = await stream.get_frame()
                if success:
                    message = {"type": "video_frame", "id": video_id, "frame": frame_base64}
                    disconnected_ws = set()
                    for ws in list(self.streaming_clients.get(video_id, set())):
                        try:
                            await ws.send_json(message)
                        except WebSocketDisconnect:
                            disconnected_ws.add(ws)
                        except Exception as e:
                            print(f"Error sending frame to {ws.client} for {video_id}: {e}")
                            disconnected_ws.add(ws)
                    
                    for ws in disconnected_ws:
                        self.disconnect(ws)
                        print(f"Client {ws.client} disconnected during frame streaming for {video_id}.")
                else:
                    print(f"Failed to get frame for {video_id}. Stream might have ended or encountered an error.")
                    # stream.is_running = False # 스트림 서비스 내부에서 상태 변경 권장
                    break 
                
                await asyncio.sleep(1 / stream.fps if stream.fps > 0 else 0.1)
        except asyncio.CancelledError:
            print(f"Video streaming task for {video_id} was cancelled.")
        finally:
            print(f"Stopped streaming frames for {video_id}")
            # 이 태스크가 취소되거나 종료될 때 스트림 세션 정리 로직은 stop_streaming_session에서 담당

    async def _analyze_video_feed(self, video_id: str, db: Session):
        stream = self.video_streams.get(video_id)
        if not stream or not stream.is_running or stream.cap is None:
            print(f"Stream for {video_id} not ready for analysis.")
            return
        
        print(f"Starting analysis for {video_id}")
        frame_interval_seconds = 5
        temp_frame_path = ""

        try:
            while stream.is_running:
                if not stream.frame_buffer:
                    await asyncio.sleep(0.5) # 버퍼 채워질 시간 확보
                    continue
                
                # 버퍼의 최신 프레임 사용
                try:
                    latest_frame_for_analysis = stream.frame_buffer[-1]
                except IndexError:
                    await asyncio.sleep(0.1) # 버퍼가 갑자기 비워진 경우
                    continue
                
                is_fire, confidence = video_processing_service.detect_fire_yolo(latest_frame_for_analysis)
                
                if is_fire and confidence >= settings.YOLO_CONFIDENCE_THRESHOLD:
                    now = time.time()
                    last_alert = self.last_alert_time.get(video_id, 0)
                    
                    if now - last_alert >= 10: # 10초 쿨다운
                        temp_frame_path = f"temp_frame_gemini_{video_id}_{int(now)}.jpg"
                        try:
                            cv2.imwrite(temp_frame_path, latest_frame_for_analysis)
                            risk_level = await analysis_service.analyze_image_with_gemini(temp_frame_path)
                            print(f"[{video_id}] Gemini analysis result: {risk_level}")
                            
                            if risk_level.lower() in ['위험', '주의']:
                                self.last_alert_time[video_id] = now
                                _, buffer = cv2.imencode('.jpg', latest_frame_for_analysis)
                                frame_base64_for_alert = base64.b64encode(buffer).decode('utf-8')
                                
                                await self.broadcast_suspect_event(
                                    video_id=video_id,
                                    timestamp=datetime.now().isoformat(),
                                    confidence=float(confidence),
                                    analysis=risk_level,
                                    frame_base64=frame_base64_for_alert
                                )
                                
                                # DB에 이벤트 저장 (겹치지 않도록 시간 등으로 추가 조건 고려 가능)
                                fire_event = models.FireEvent(
                                    video_id=video_id,
                                    timestamp=datetime.now(),
                                    event_type=risk_level,
                                    confidence=float(confidence),
                                    analysis=f"Gemini: {risk_level}, YOLO confidence: {confidence:.2f}"
                                )
                                db.add(fire_event)
                                db.commit()
                                print(f"[{video_id}] Fire event (type: {risk_level}) saved to DB.")
                        except Exception as e:
                            print(f"[{video_id}] Error during Gemini analysis or alert: {e}")
                        finally:
                            if os.path.exists(temp_frame_path):
                                os.remove(temp_frame_path)
                                temp_frame_path = "" # 경로 초기화
                    else:
                        # print(f"[{video_id}] Alert cooldown for Gemini. Skipping.")
                        pass 
                
                await asyncio.sleep(frame_interval_seconds)
        except asyncio.CancelledError:
            print(f"Video analysis task for {video_id} was cancelled.")
        finally:
            print(f"Stopped analysis for {video_id}")
            if os.path.exists(temp_frame_path): # 최종적으로 임시 파일 삭제 확인
                os.remove(temp_frame_path)

    async def subscribe_to_video_stream(self, websocket: WebSocket, video_id: str, db: Session):
        video_obj = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video_obj or not video_obj.filename:
            await websocket.send_json({"error": f"Video with id {video_id} not found or filename missing."})
            return

        video_file_path = os.path.join(settings.VIDEO_UPLOAD_DIR, video_obj.filename)
        if not os.path.exists(video_file_path):
            await websocket.send_json({"error": f"Video file for {video_id} not found on server."})
            video_obj.status = "Error: File missing"
            db.commit()
            return

        await self.start_streaming_session(video_id, video_file_path, db)
        if video_id in self.streaming_clients:
            self.streaming_clients[video_id].add(websocket)
            print(f"Client {websocket.client} subscribed to video stream {video_id}. Total: {len(self.streaming_clients[video_id])}")
        
        await websocket.send_json({"status": "subscribed", "video_id": video_id})

    async def unsubscribe_from_video_stream(self, websocket: WebSocket, video_id: str):
        if video_id in self.streaming_clients and websocket in self.streaming_clients[video_id]:
            self.streaming_clients[video_id].remove(websocket)
            print(f"Client {websocket.client} unsubscribed from video stream {video_id}. Remaining: {len(self.streaming_clients[video_id])}")
            # if not self.streaming_clients[video_id]:
            #     # 모든 클라이언트가 떠났을 때 스트림 자동 중지 로직 (start_streaming_session과 대칭적으로)
            #     # db_session = next(get_db()) # 새 세션 필요
            #     # await self.stop_streaming_session(video_id, db_session)
            #     # db_session.close()
            #     print(f"No more clients for {video_id}. Stream will stop if auto-stop is active.")
        else:
             print(f"Client {websocket.client} tried to unsubscribe from {video_id} but was not subscribed or video not streaming.")

    async def stop_streaming_session(self, video_id: str, db: Session):
        print(f"Attempting to stop streaming session for {video_id}")
        
        analysis_task = self.analysis_tasks.pop(video_id, None)
        if analysis_task and not analysis_task.done():
            analysis_task.cancel()
            try: await analysis_task
            except asyncio.CancelledError: print(f"Analysis task for {video_id} cancelled.")
            except Exception as e: print(f"Error during analysis task cancellation for {video_id}: {e}")

        streaming_task = self.streaming_tasks.pop(video_id, None)
        if streaming_task and not streaming_task.done():
            streaming_task.cancel()
            try: await streaming_task
            except asyncio.CancelledError: print(f"Streaming task for {video_id} cancelled.")
            except Exception as e: print(f"Error during streaming task cancellation for {video_id}: {e}")
        
        stream_service = self.video_streams.pop(video_id, None)
        if stream_service:
            stream_service.release()
            print(f"StreamingService object released for {video_id}")

        if video_id in self.streaming_clients:
            del self.streaming_clients[video_id]
            print(f"Streaming client list cleared for {video_id}")
        
        video_obj = db.query(models.Video).filter(models.Video.id == video_id).first()
        if video_obj and video_obj.status == "Streaming": # 스트리밍 중이었을 때만 상태 변경
            video_obj.status = "Stopped"
            db.commit()
            print(f"Video {video_id} status updated to 'Stopped' in DB.")
        
        print(f"Streaming session for {video_id} fully stopped and cleaned up.")

connection_manager = UnifiedConnectionManager()
