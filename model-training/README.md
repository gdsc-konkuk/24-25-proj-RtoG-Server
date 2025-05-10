# 모델 학습 (Model Training)

이 프로젝트는 화재 감지를 위한 YOLO 모델 학습 및 관리 코드를 포함합니다.

## 기능

- YOLO 모델 학습 및 미세조정
- 모델 검증 및 평가
- 모델 배포 준비

## 설치 및 실행

1. 가상환경 활성화:
```bash
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 모델 학습:
```bash
# YOLO 모델 학습 예시
python -m ultralytics.yolo.v8.detect.train data=yolo_custom.yaml model=yolov8n.pt epochs=100 imgsz=640
```

## 디렉토리 구조

- `training/`: 학습된 모델 파일과 가중치
- `runs/`: 학습 로그 및 결과물 