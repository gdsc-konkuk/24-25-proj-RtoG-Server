# server/config.py
# 애플리케이션 설정 관리
# - 환경 변수 로드
# - 기본 설정값 정의
# - API 및 데이터베이스 설정

from typing import List
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "RTOG API Server"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    OPENAPI_URL: str = "/api/openapi.json"
    DOCS_URL: str = "/api/docs"
    REDOC_URL: str = "/api/redoc"
    
    VIDEO_STORAGE_PATH: str = "static/video"  # 원본 비디오 저장 경로
    RECORD_STORAGE_PATH: str = "static/record"  # 이벤트 영상과 썸네일 저장 경로
    ALLOWED_VIDEO_TYPES: List[str] = ["video/mp4", "video/x-msvideo", "video/quicktime"]
    MAX_VIDEO_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    YOLO_MODEL_PATH: str = "best.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    SQLALCHEMY_DATABASE_URL: str = "sqlite:///./database.db"
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()