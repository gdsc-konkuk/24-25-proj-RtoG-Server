# 화재 감지 API 서버

실시간 화재 감지를 위한 API 서버 모듈입니다. YOLO 모델을 사용한 화재 감지와 Gemini AI를 활용한 위험도 분석 기능을 제공합니다.

## 주요 기능

- **실시간 비디오 스트리밍**: WebSocket을 통한 CCTV 영상 실시간 전송
- **화재 감지**: YOLOv8 모델을 통한 영상 내 화재 감지
- **위험도 분석**: Google Gemini AI를 활용한 화재 상황 위험도 분석
- **이벤트 관리**: 감지된 화재 이벤트 기록 및 알림 기능

## 기술 스택

- **FastAPI**: 고성능 웹 API 프레임워크
- **WebSocket**: 실시간 양방향 통신
- **SQLAlchemy**: ORM 데이터베이스 인터페이스
- **YOLOv8**: 화재 감지를 위한 객체 탐지 모델
- **Google Gemini**: 화재 위험도 분석을 위한 AI 모델

## 설치 방법

1. 저장소 복제
   ```bash
   git clone <repository_url>
   cd api-server
   ```

2. 가상환경 설정 및 의존성 설치
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. 환경 변수 설정
   ```bash
   # .env 파일 생성
   touch .env
   ```
   
   `.env` 파일에 다음 내용 추가:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 실행 방법

1. 서버 실행
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. API 문서 접속
   - Swagger UI: http://localhost:8000/api/docs
   - ReDoc: http://localhost:8000/api/redoc

## API 엔드포인트

### REST API
- `GET /api/videos`: 모든 비디오 목록 조회
- `GET /api/suspects/{id}/events`: 특정 비디오의 화재 감지 이벤트 조회

### WebSocket
- `WebSocket /ws/suspects`: 화재 감지 알림 구독
- `WebSocket /ws/stream`: 실시간 비디오 스트림 구독

## 모듈 구조

- **main.py**: FastAPI 애플리케이션 및 핵심 로직
- **models.py**: SQLAlchemy 데이터베이스 모델
- **schemas.py**: Pydantic 스키마 정의
- **database.py**: 데이터베이스 연결 및 세션 관리
- **gemini.py**: Google Gemini AI 연동 모듈
- **yolo11n.pt**: 화재 감지를 위한 사전 훈련된 YOLOv8 모델

## 클라이언트 연결 예시

### WebSocket 화재 감지 알림 구독
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/suspects');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('화재 감지:', data);
};
```

### WebSocket 비디오 스트림 구독
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');
ws.onopen = () => {
  ws.send(JSON.stringify({ action: 'subscribe', video_id: 'video_1' }));
};
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // 비디오 프레임 처리
};
```

## 필요 조건

- Python 3.8 이상
- YOLO 모델 파일 (yolo11n.pt)
- Google Gemini API 키 