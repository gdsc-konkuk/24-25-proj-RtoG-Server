# api-server/database.py
# 이 파일은 SQLAlchemy를 사용하여 데이터베이스 연결 및 세션 관리를 설정합니다.
# config.py에 정의된 데이터베이스 URL을 사용하여 데이터베이스 엔진(engine)을 생성하고,
# 세션 생성을 위한 SessionLocal 클래스와 데이터베이스 모델의 기반이 되는 Base 클래스를 정의합니다.
# 또한, FastAPI의 의존성 주입 시스템을 통해 라우터에서 데이터베이스 세션을 사용할 수 있도록 하는
# get_db 함수를 제공합니다.

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"

engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 