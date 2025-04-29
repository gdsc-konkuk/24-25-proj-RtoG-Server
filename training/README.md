# YOLOv11을 이용한 연기 감지 모델 학습

이 프로젝트는 YOLOv11을 사용하여 연기(smoke) 감지 모델을 학습하는 코드를 제공합니다. 데이터셋은 [AI For Mankind](https://aiformankind.org/)에서 제공한 Wildfire Smoke 데이터셋을 사용합니다.

## 데이터셋 정보

- 클래스: 1개 (연기 - 'smoke')
- 이미지 수: 737개
- 형식: YOLOv11 형식

## 환경 설정

필수 패키지 설치:

```bash
pip install -r requirements.txt
```

## 모델 학습하기

다음 명령어로 YOLOv11 모델 학습을 시작합니다:

```bash
python train_yolov11.py
```

기본적으로 모델은 다음 설정으로 학습됩니다:
- 에폭: 100
- 배치 크기: 16
- 이미지 크기: 640
- 최적화 알고리즘: AdamW

학습 과정 중 최상의 모델은 `yolov11_smoke_detection/train_results/weights/` 디렉토리에 저장됩니다.

## 예측 실행하기

학습된 모델을 사용하여 새 이미지나 비디오에서 연기를 감지하려면:

```bash
python predict.py --source [이미지 또는 비디오 경로] --conf 0.25
```

매개변수:
- `--model`: 모델 파일 경로 (기본값: yolov11_smoke_detection/train_results/weights/best.pt)
- `--source`: 예측할 이미지, 비디오 또는 디렉토리 경로
- `--conf`: 감지 신뢰도 임계값 (기본값: 0.25)
- `--save`: 결과 저장 여부 (기본값: True)

## 모델 성능

모델 검증은 학습 과정에서 자동으로 수행됩니다. 성능 메트릭은 콘솔에 출력됩니다.

## 참고 사항

- 더 좋은 성능을 위해 학습 매개변수를 조정할 수 있습니다.
- `train_yolov11.py` 파일에서 모델 크기(n, s, m, l, x)를 변경할 수 있습니다.
- 학습 속도를 높이기 위해 GPU 사용을 권장합니다. 