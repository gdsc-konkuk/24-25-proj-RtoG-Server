# 데이터셋 처리 및 학습 파일 추가 PR

## 변경 내용 요약
이 PR에서는 다음과 같은 작업을 수행했습니다:
- YOLOv11 기반 학습을 위한 데이터셋 처리 파일 추가
- 학습용 이미지와 라벨 데이터 추가
- 학습 스크립트 구현

## 추가된 파일
- 학습 스크립트: `training/train_yolov11.py`
- 학습 데이터: 
  - 훈련 이미지 (1000+): `training/train/images/`
  - 훈련 라벨: `training/train/labels/`
  - 검증 이미지 (100+): `training/valid/images/`
  - 검증 라벨: `training/valid/labels/`
- 사전 훈련된 모델: `training/yolo11n.pt`

## 테스트 방법
1. 다음 명령어로 학습 스크립트 실행:
   ```bash
   python training/train_yolov11.py
   ```
2. 학습 결과는 `training/runs/` 디렉토리에 저장됩니다.

## 관련 이슈
- 데이터셋 구축 및 학습 파이프라인 구현과 관련된 이슈

## 기타 참고사항
- 데이터셋은 이미지 분할 및 증강 작업을 통해 확장되었습니다.
- 사전 훈련된 모델을 기반으로 추가 학습이 가능합니다. 