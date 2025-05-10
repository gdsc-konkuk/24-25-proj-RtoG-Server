from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "RTOG API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    OPENAPI_URL: str = "/openapi.json"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000"]  # React frontend
    
    # Video settings
    VIDEO_UPLOAD_DIR: str = "uploads"
    MAX_VIDEO_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/quicktime"]
    
    class Config:
        case_sensitive = True

settings = Settings() 