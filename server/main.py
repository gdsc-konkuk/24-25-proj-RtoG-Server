# server/main.py
# FastAPI 애플리케이션의 메인 진입점
# - 앱 설정, 미들웨어 구성, 라우터 등록
# - 데이터베이스 초기화 및 앱 시작/종료 이벤트 처리

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from config import settings
from database import engine, Base, get_db
from routers import records as record_router
from routers import websockets as websocket_router
from routers import lives as live_router
from websocket_manager import connection_manager

# 개발용: 데이터베이스 테이블 재생성
print("Dropping all tables for recreation...")
Base.metadata.drop_all(bind=engine)
print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Database tables created successfully.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    RTOG 화재 감지 서버 API.
    
    ## 주요 기능
    * 실시간 CCTV 스트리밍 및 화재 감지 (Live)
    * 화재 이벤트 기록 조회 (Records)
    * WebSocket을 통한 실시간 스트리밍 및 알림
    """,
    openapi_url=settings.OPENAPI_URL,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 라우터 등록
app.include_router(live_router.router, prefix=settings.API_V1_STR + "/lives", tags=["Live"])
app.include_router(record_router.router, prefix=settings.API_V1_STR + "/records", tags=["Records"])
app.include_router(websocket_router.router, tags=["WebSockets"])

@app.on_event("startup")
async def startup_event():
    print(f"Application startup complete. YOLO model loaded: {settings.YOLO_MODEL_PATH}")
    print(f"Uploads directory: {settings.VIDEO_UPLOAD_DIR}")
    if not settings.GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY is not set in .env file or config.")
    else:
        print("Gemini API key is configured.")

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutting down...")
    db_session_for_shutdown: Session = next(get_db())
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

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} v{settings.VERSION}"}

# uvicorn main:app --reload 등으로 실행 