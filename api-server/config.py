# api-server/config.py
# 이 파일은 Pydantic의 BaseSettings를 사용하여 애플리케이션 전체의 설정을 관리합니다.
# 환경 변수 또는 .env 파일에서 설정을 로드하며, 프로젝트 이름, API 버전, 데이터베이스 URL,
# 비디오 처리 관련 설정, CORS 설정, 외부 API 키 (예: Gemini) 등 다양한 설정을 포함합니다.
# settings 객체를 통해 다른 모듈에서 이 설정 값들을 쉽게 가져다 사용할 수 있습니다.

from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    PROJECT_NAME: str = "RTOG API"  # 기본값, 추후 main.py의 내용으로 업데이트 가능
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1" # app/core/config.py 에 있었음.
    OPENAPI_URL: str = "/api/openapi.json" # main.py 에서 가져올 예정
    DOCS_URL: str = "/api/docs" # main.py 에서 가져올 예정
    REDOC_URL: str = "/api/redoc" # main.py 에서 가져올 예정
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "*"] # 양쪽의 것을 통합
    
    # Video settings from app/core/config.py
    VIDEO_UPLOAD_DIR: str = "uploads"
    MAX_VIDEO_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_VIDEO_TYPES: List[str] = ["video/mp4", "video/quicktime"]

    # Database URL from database.py (or main.py if it was there)
    SQLALCHEMY_DATABASE_URL: str = "sqlite:///./database.db"

    # YOLO Model Path from main.py
    YOLO_MODEL_PATH: str = "training/yolo11n.pt" # main.py 에서 가져올 예정
    YOLO_CONFIDENCE_THRESHOLD: float = 0.1 # main.py 에서 가져올 예정

    # Gemini API Key - .env 파일에서 로드하도록 gemini.py에 있었으나, 설정 클래스에서도 관리 가능
    GEMINI_API_KEY: Optional[str] = None # .env 파일에서 로드하는 것을 권장

    class Config:
        env_file = ".env" # .env 파일 사용 명시
        env_file_encoding = 'utf-8'
        case_sensitive = True

settings = Settings() 