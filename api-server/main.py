# api-server/main.py
# 이 파일은 FastAPI 애플리케이션의 메인 진입점(entrypoint)입니다.
# FastAPI 앱 인스턴스를 생성하고, 전역 설정을 로드하며, 미들웨어(예: CORS)를 구성합니다.
# 또한, 애플리케이션 시작(startup) 및 종료(shutdown) 시 수행할 작업을 정의하고,
# routers 모듈에 정의된 HTTP 및 WebSocket 라우터들을 앱에 포함시켜 API 엔드포인트를 제공합니다.
# 데이터베이스 초기화 로직 (개발용) 및 기본적인 상태 확인 엔드포인트도 포함되어 있습니다.
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import asyncio

# 설정, DB, 모델, 스키마, 서비스, 라우터 등 import
from .config import settings
from .database import engine, Base, get_db
from . import models, schemas
from .services import video_processing_service, analysis_service
from .routers import videos as video_router
from .routers import websockets as websocket_router
from .websockets import connection_manager

# 애플리케이션 시작 시 데이터베이스 테이블 생성
# 실제 프로덕션에서는 Alembic과 같은 마이그레이션 도구 사용을 권장
print("Dropping all tables for recreation (if any)... This should be handled by migrations in production.")
Base.metadata.drop_all(bind=engine) # 개발 중에는 테이블을 매번 재생성 (데이터 유지 안됨)
print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Database tables created successfully.")

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
     통합된 RTOG API 서버.
    
    ## 주요 기능
    * 비디오 업로드 및 정보 관리 (HTTP)
    * 비디오 실시간 스트리밍 (WebSocket)
    * 화재 감지 (YOLO) 및 위험 분석 (Gemini) 후 알림 (WebSocket)
    
    ## API 엔드포인트
    * HTTP API: `/api/v1/videos` (VideoRouter 참조)
    * WebSocket API: `/ws/suspects`, `/ws/stream/{video_id}` (WebSocketRouter 참조)
    """,
    openapi_url=settings.OPENAPI_URL,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1} # 스키마 모델 기본적으로 닫기
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"], # 모든 HTTP 메소드 허용
    allow_headers=["*"] # 모든 HTTP 헤더 허용
)

# HTTP 라우터 포함
app.include_router(video_router.router, prefix=settings.API_V1_STR + "/videos", tags=["Videos"])

# WebSocket 라우터 포함 (prefix 없이 루트에서 직접 접근)
app.include_router(websocket_router.router, tags=["WebSockets"])

@app.on_event("startup")
async def startup_event():
    # 애플리케이션 시작 시 필요한 작업 (예: 모델 로드, 디렉토리 생성 등)
    # VideoProcessingService 생성자에서 YOLO 모델 로드 및 uploads 디렉토리 생성됨
    # Gemini API는 gemini.py 로드 시 설정됨
    print(f"Application startup complete. YOLO model loaded: {settings.YOLO_MODEL_PATH}")
    print(f"Uploads directory: {settings.VIDEO_UPLOAD_DIR}")
    if not settings.GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY is not set in .env file or config. Gemini analysis will fail.")
    else:
        print("Gemini API key is configured.")
    
    # 테스트용/기본 비디오 DB에 추가 (선택 사항)
    # db: Session = next(get_db()) # 임시 세션
    # try:
    #     # 여기에 기본 비디오 정보 추가 로직
    #     # 예: test_video = db.query(models.Video).filter(models.Video.filename == "sample.mp4").first()
    #     # if not test_video and os.path.exists(os.path.join(settings.VIDEO_UPLOAD_DIR, "sample.mp4")):
    #     #     # ... models.Video 객체 만들고 db.add(), db.commit() ...
    #     #     print("Added/verified sample video in DB.")
    # finally:
    #     db.close()

@app.on_event("shutdown")
async def shutdown_event():
    # 애플리케이션 종료 시 실행되어야 하는 정리 작업
    print("Application shutting down...")
    # 활성화된 모든 스트리밍 세션 정리
    db_session_for_shutdown: Session = next(get_db()) # 새 세션
    try:
        active_video_ids = list(connection_manager.video_streams.keys())
        if active_video_ids:
            print(f"Stopping {len(active_video_ids)} active streaming session(s): {active_video_ids}")
            for video_id in active_video_ids:
                await connection_manager.stop_streaming_session(video_id, db_session_for_shutdown)
        else:
            print("No active streaming sessions to stop.")
    except Exception as e:
        print(f"Error during shutdown stream cleanup: {e}")
    finally:
        db_session_for_shutdown.close()
    print("Application shutdown complete.")

@app.get("/", summary="Root endpoint for API health check")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} v{settings.VERSION}"}

# uvicorn main:app --reload 등으로 실행 