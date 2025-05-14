# 화재 감지 API 서버

실시간 화재 감지를 위한 API 서버 모듈입니다. YOLO 모델을 사용한 화재 감지와 Gemini AI를 활용한 위험도 분석 기능을 제공합니다.

## 주요 기능

- **비디오 관리**: HTTP를 통한 비디오 업로드, 정보 조회, 처리 요청
- **실시간 비디오 스트리밍**: WebSocket을 통한 CCTV 영상 실시간 전송
- **화재 감지**: YOLO 모델을 통한 영상 내 화재 감지
- **위험도 분석**: Google Gemini AI를 활용한 화재 상황 위험도 분석
- **이벤트 알림 및 관리**: 감지된 화재 이벤트 기록 및 WebSocket을 통한 실시간 알림

## 기술 스택

- **FastAPI**: 고성능 웹 API 프레임워크
- **WebSocket**: 실시간 양방향 통신
- **SQLAlchemy**: ORM 데이터베이스 인터페이스
- **Pydantic**: 데이터 유효성 검사 및 설정 관리
- **YOLO**: 화재 감지를 위한 객체 탐지 모델 (예: YOLOv8)
- **Google Gemini**: 화재 위험도 분석을 위한 AI 모델

## 설치 방법

1.  저장소 복제
    ```bash
    git clone <repository_url>
    cd server
    ```

2.  가상환경 설정 및 의존성 설치
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  환경 변수 설정
    ```bash
    # .env 파일 생성 (server 디렉토리 내)
    touch .env
    ```

    `.env` 파일에 다음 내용 추가 (필요에 따라 값 수정):
    ```env
    # .env 예시
    PROJECT_NAME="RTOG API Server"
    VERSION="1.0.0"
    
    # 개발용 SQLite DB 경로
    SQLALCHEMY_DATABASE_URL="sqlite:///./database.db"
    
    # YOLO 모델 경로 (프로젝트 루트 기준 또는 절대 경로)
    YOLO_MODEL_PATH="training/yolo11n.pt"
    YOLO_CONFIDENCE_THRESHOLD=0.1
    
    # Gemini API 키
    GEMINI_API_KEY="your_gemini_api_key_here"
    
    # 비디오 업로드 디렉토리 (main.py 기준 상대 경로)
    VIDEO_UPLOAD_DIR="uploads"
    MAX_VIDEO_SIZE=104857600 # 100MB
    # ALLOWED_VIDEO_TYPES='["video/mp4", "video/quicktime"]' # 리스트는 문자열로
    ```
    *주의*: `ALLOWED_VIDEO_TYPES` 와 같은 리스트 형태의 환경 변수는 애플리케이션 코드에서 적절히 파싱해야 할 수 있습니다. `config.py`의 `BaseSettings`는 기본적으로 JSON 문자열을 파싱할 수 있습니다.

## 실행 방법

1.  서버 실행 (`server` 디렉토리 내에서 실행)
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

2.  API 문서 접속
    - Swagger UI: http://localhost:8000/api/docs
    - ReDoc: http://localhost:8000/api/redoc

## 도커로 실행

1. 도커 이미지 가져오기
    ```bash
    docker pull peter012677/rtog-server
    ```

2. 도커 실행 (기본)
    ```bash
    docker run -d \
      -p 8000:8000 \
      -e GEMINI_API_KEY=your_api_key_here \
      --name rtog-server \
      peter012677/rtog-server
    ```

3. 파일시스템 마운트 옵션 (영속성 보전) : 꼭 사용할 필요x
    ```bash
    docker run -d \
      -p 8000:8000 \
      -e GEMINI_API_KEY=your_api_key_here \
      -v $(pwd)/database.db:/app/database.db \
      -v $(pwd)/static:/app/static \
      --name rtog-server \
      peter012677/rtog-server
    ```

## API 엔드포인트

### HTTP API (Prefix: `/api/v1`)

-   **Videos (`/videos`)**
    -   `POST /upload`: 새 비디오 업로드.
        -   Request Body: `UploadFile` (form-data)
        -   Response: `schemas.Video`
    -   `GET /process/{video_id}`: 특정 비디오 처리 요청 (예: 좌표 추출 - 현재는 단순 예시 기능).
        -   Response: `dict` (처리 결과)
    -   `GET /`: 모든 비디오 목록 조회.
        -   Response: `List[schemas.Video]`
    -   `GET /{video_id}`: 특정 비디오 정보 조회.
        -   Response: `schemas.Video`
    -   `GET /{video_id}/events`: 특정 비디오의 화재 이벤트 조회.
        -   Response: `List[schemas.FireEvent]`

### WebSocket API

-   `GET /ws/suspects?owner_id={owner_id}`: 화재 및 특이사항 감지 실시간 알림 구독.
    -   Query Parameters:
        - `owner_id`: (필수) CCTV 소유자 ID
    -   서버가 전송하는 메시지 예시:
        ```json
        {
          "event": "fire_detected",
          "data": {
            "id": "video_id_example",
            "timestamp": "2023-10-27T10:30:00Z",
            "confidence": 0.85,
            "analysis": "위험", // Gemini 분석 결과
            "frame": "base64_encoded_image_string"
          }
        }
        ```
-   `GET /ws/stream/{video_id}?owner_id={owner_id}`: 특정 비디오(`video_id`)에 대한 실시간 스트리밍 구독.
    -   Path Parameters:
        - `video_id`: 스트리밍할 비디오 ID
    -   Query Parameters:
        - `owner_id`: (필수) CCTV 소유자 ID
    -   서버가 전송하는 메시지 예시:
        ```json
        {
          "type": "video_frame",
          "id": "video_id_example",
          "frame": "base64_encoded_jpeg_image_string"
        }
        ```

## 모듈 구조 (`server/`)

-   **main.py**: FastAPI 애플리케이션 인스턴스 생성, 설정 로드, 미들웨어 구성, 라우터 포함, 시작/종료 이벤트 처리.
-   **config.py**: Pydantic `BaseSettings`를 사용한 애플리케이션 전체 설정 관리.
-   **database.py**: SQLAlchemy를 사용한 데이터베이스 연결 및 세션 관리.
-   **models.py**: SQLAlchemy 데이터베이스 모델(테이블) 정의.
-   **schemas.py**: API 요청/응답 데이터 유효성 검사 및 직렬화를 위한 Pydantic 스키마 정의.
-   **services.py**: 핵심 비즈니스 로직을 포함하는 서비스 클래스들.
    -   `VideoProcessingService`: 비디오 파일 처리, YOLO 감지 등.
    -   `AnalysisService`: Gemini API를 이용한 분석 로직.
    -   `StreamingService`: 비디오 프레임 스트리밍 로직.
-   **routers/**: API 엔드포인트를 정의하는 라우터 모듈.
    -   `videos.py`: 비디오 관련 HTTP API 라우터.
    -   `websockets.py`: WebSocket 관련 API 라우터.
-   **websockets.py**: `UnifiedConnectionManager` 클래스를 통해 WebSocket 연결 관리 및 메시지 브로드캐스팅 로직 처리.
-   **gemini.py**: Google Gemini API 연동 모듈.
-   **uploads/**: 업로드된 비디오 파일 저장 디렉토리 (설정에 따라 변경 가능).
-   **training/yolo11n.pt** (또는 `settings.YOLO_MODEL_PATH` 경로의 파일): 화재 감지를 위한 사전 훈련된 YOLO 모델.

## 클라이언트 연결 예시

### WebSocket 화재 감지 알림 구독
```javascript
const owner_id = 'your_owner_id_here'; // 소유자 ID
const suspectsWs = new WebSocket(`ws://localhost:8000/ws/suspects?owner_id=${owner_id}`);

suspectsWs.onopen = () => {
  console.log('Connected to suspect alerts WebSocket.');
};

suspectsWs.onmessage = (event) => {
  try {
    const data = JSON.parse(event.data);
    console.log('화재/특이사항 감지:', data);
    if (data.event === 'fire_detected' && data.data.frame) {
      console.log('Frame received for video:', data.data.id);
    }
  } catch (error) {
    console.error('Error parsing suspect event message:', error);
  }
};

suspectsWs.onclose = (event) => {
  if (event.code === 4003) {
    console.log('Unauthorized access. Please check your owner ID.');
  } else {
    console.log('Disconnected from suspect alerts WebSocket.');
  }
};

suspectsWs.onerror = (error) => {
  console.error('Suspect alerts WebSocket error:', error);
};
```

### WebSocket 비디오 스트림 구독
```javascript
const videoId = 'your_video_id_here'; // 실제 비디오 ID로 대체
const owner_id = 'your_owner_id_here'; // 소유자 ID
const streamWs = new WebSocket(`ws://localhost:8000/ws/stream/${videoId}?owner_id=${owner_id}`);

streamWs.onopen = () => {
  console.log(`Connected to video stream for ${videoId}.`);
};

streamWs.onmessage = (event) => {
  try {
    const data = JSON.parse(event.data);
    if (data.type === 'video_frame' && data.id === videoId && data.frame) {
      console.log(`Frame received for video ${data.id}`);
    } else if (data.error) {
      console.error(`Server error for video stream ${videoId}:`, data.error);
      streamWs.close();
    } else if (data.status === 'subscribed') {
      console.log(`Successfully subscribed to video stream: ${data.video_id}`);
    }
  } catch (error) {
    console.error('Error parsing video stream message:', error);
  }
};

streamWs.onclose = (event) => {
  if (event.code === 4003) {
    console.log('Unauthorized access. Please check your owner ID.');
  } else if (event.code === 4004) {
    console.log('Video not found.');
  } else {
    console.log(`Disconnected from video stream for ${videoId}.`);
  }
};

streamWs.onerror = (error) => {
  console.error(`Video stream WebSocket error for ${videoId}:`, error);
};
```