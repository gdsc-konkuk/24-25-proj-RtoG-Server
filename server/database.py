# server/database.py
# SQLAlchemy를 사용한 데이터베이스 연결 및 세션 관리
# - 데이터베이스 엔진 생성
# - 세션 관리
# - 데이터베이스 초기화

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
from pathlib import Path
from config import settings

# SQLAlchemy 엔진 생성
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 세션 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스 생성
Base = declarative_base()

# 데이터베이스 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_database():
    """서버 시작 시 데이터베이스 초기화"""
    print("데이터베이스 초기화 시작...")
    db = next(get_db())
    try:
        # 메타데이터 파일 읽기
        metadata_path = Path("static/metadata/videos.json")
        if not metadata_path.exists():
            print("Warning: Metadata file not found")
            return
        
        print(f"메타데이터 파일 읽기: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"메타데이터에서 {len(metadata['videos'])}개의 비디오 정보를 찾았습니다.")
        
        # 각 비디오 메타데이터를 DB에 저장
        from models import Video  # 여기서 import
        for video_metadata in metadata["videos"]:
            video_id = video_metadata["id"]
            print(f"비디오 처리 중: {video_id}")
            
            # 기존 비디오 정보 조회
            existing_video = db.query(Video).filter(Video.id == video_id).first()
            
            if existing_video:
                # 기존 정보 업데이트
                print(f"기존 비디오 정보 업데이트: {video_id}")
                for key, value in video_metadata.items():
                    setattr(existing_video, key, value)
            else:
                # 새 비디오 정보 생성
                print(f"새 비디오 정보 생성: {video_id}")
                db_video = Video(**video_metadata)
                db.add(db_video)
        
        db.commit()
        print("데이터베이스 초기화 완료")
        
    except Exception as e:
        print(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
        db.rollback()
    finally:
        db.close() 