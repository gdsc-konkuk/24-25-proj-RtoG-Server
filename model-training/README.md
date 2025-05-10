# YOLOv11을 이용한 연기 감지 모델 학습

이 모듈은 YOLOv11을 사용하여 화재 및 연기 감지 모델을 학습하는 코드를 제공합니다.

## 주요 파일 설명

- `train_yolov11.py`: YOLOv11 모델 학습 스크립트
- `predict.py`: 학습된 모델을 사용한 예측 스크립트
- `yolo_custom.yaml`: YOLOv11 데이터 설정 파일 (화재/연기 감지용)
- `yolo11n.pt`: YOLOv11 사전 학습 모델

## 환경 설정

### 1. 가상환경 활성화
```bash
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 2. 의존성 설치

#### CPU 버전:
```bash
pip install -r requirements.txt
```

#### GPU(CUDA) 버전:
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```
> ⚠️ CUDA 버전(cu118)은 사용하는 NVIDIA 드라이버에 맞게 수정해야 할 수 있습니다.

## 데이터셋 설정

### 중요: yolo_custom.yaml 파일 경로 수정

모델 학습 전 `yolo_custom.yaml` 파일의 경로를 **반드시 수정**해야 합니다:

```yaml
# 상대경로로 변경
path: ../processed  # 프로젝트 루트 기준 processed 디렉토리
```

이 경로는 실제 데이터셋이 위치한 곳을 가리키도록 변경해야 합니다.

## 파일 사용법

### 1. 모델 학습 (train_yolov11.py)

train_yolov11.py는 YOLOv11 모델을 처음부터 학습하거나 전이학습하는 스크립트입니다.

```bash
# 기본 학습
python train_yolov11.py

# data.yaml 파일이 없어 오류가 발생할 경우 다음과 같이 코드 수정 필요:
# data_yaml_path = "yolo_custom.yaml" (train_yolov11.py의 13번 라인 수정)
```

학습 매개변수 (train_yolov11.py에서 직접 수정 가능):
- 에폭: 100
- 배치 크기: 16
- 이미지 크기: 640
- 최적화 알고리즘: AdamW

### 2. 예측 실행 (predict.py)

학습된 모델로 새 이미지나 비디오에서 연기와 화재를 감지합니다.

```bash
# 기본 사용법
python predict.py --source [이미지 또는 비디오 경로]

# 고급 옵션
python predict.py --model [모델 경로] --source [소스 경로] --conf 0.25 --save
```

매개변수:
- `--model`: 모델 파일 경로 (기본값: yolov11_smoke_detection/train_results/weights/best.pt)
- `--source`: 예측할 이미지, 비디오 또는 디렉토리 경로 (필수)
- `--conf`: 감지 신뢰도 임계값 (기본값: 0.25)
- `--save`: 결과 저장 여부 (기본값: True)

## 클래스 정보

`yolo_custom.yaml` 파일에 정의된 클래스:
```
0: 흑색연기
1: 백색/회색연기
2: 화염
3: 구름
4: 안개/연무
5: 굴뚝연기
```

## 결과 확인

- 학습 결과: `yolov11_smoke_detection/train_results/` 디렉토리
- 학습된 모델: `yolov11_smoke_detection/train_results/weights/best.pt`
- 예측 결과: `yolov11_smoke_detection/predictions/` 디렉토리

## 참고 사항

- 더 나은 성능을 위해 학습 매개변수를 조정할 수 있습니다.
- 학습 속도 향상을 위해 GPU 사용을 권장합니다.
- 대용량 데이터셋 처리 시 메모리 문제가 발생할 경우, 배치 크기를 줄이세요. 