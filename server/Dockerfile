FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# static 디렉토리와 하위 디렉토리 생성
RUN mkdir -p static/video static/record

# 포트 설정
EXPOSE 8000

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV GEMINI_API_KEY=""

# 실행 명령
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]