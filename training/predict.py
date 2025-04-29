#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from ultralytics import YOLO

def predict_with_yolov11(model_path, source, conf=0.25, save=True):
    """
    학습된 YOLOv11 모델을 사용하여 이미지나 비디오에서 연기를 감지합니다.
    
    Args:
        model_path (str): 학습된 모델 파일의 경로
        source (str): 예측할 이미지, 비디오 또는 디렉토리 경로
        conf (float): 신뢰도 임계값 (0-1 사이 값)
        save (bool): 결과 저장 여부
    
    Returns:
        results: 예측 결과
    """
    # 모델 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    model = YOLO(model_path)
    print(f"모델을 로드했습니다: {model_path}")
    
    # 소스 확인
    if not os.path.exists(source):
        raise FileNotFoundError(f"소스 파일 또는 디렉토리를 찾을 수 없습니다: {source}")
    
    # 예측 실행
    print(f"다음에 대한 예측 시작: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        project="yolov11_smoke_detection",
        name="predictions",
        exist_ok=True
    )
    
    print(f"예측 완료")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 모델을 사용하여 연기 감지")
    parser.add_argument("--model", type=str, default="yolov11_smoke_detection/train_results/weights/best.pt",
                        help="학습된 모델 경로")
    parser.add_argument("--source", type=str, required=True,
                        help="예측할 이미지, 비디오 또는 디렉토리 경로")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="감지 신뢰도 임계값")
    parser.add_argument("--save", action="store_true", default=True, 
                        help="결과 저장 여부")
    
    args = parser.parse_args()
    
    predict_with_yolov11(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        save=args.save
    ) 