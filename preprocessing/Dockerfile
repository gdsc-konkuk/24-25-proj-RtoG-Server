FROM python:3.9-slim

WORKDIR /app

# OpenCV 종속성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 의존성 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 스크립트 복사
COPY image_preprocessing.py .

# 데이터 및 결과 디렉토리 생성
RUN mkdir -p /app/data/images /app/data/labels /app/processed/images /app/processed/labels

# 볼륨 설정 - 데이터와 처리 결과를 호스트와 공유
VOLUME ["/app/data", "/app/processed"]

# 엔트리포인트 설정 (경로는 컨테이너 내부 경로로 고정)
ENTRYPOINT ["python", "image_preprocessing.py", "--input-dir", "/app/data", "--output-dir", "/app/processed"] 