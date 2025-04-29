#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import torch
from ultralytics import YOLO

def train_yolov11():
    # 현재 작업 디렉토리 확인
    print("현재 작업 디렉토리:", os.getcwd())
    
    # data.yaml 파일 경로
    data_yaml_path = "data.yaml"
    
    # data.yaml 파일 읽기
    with open(data_yaml_path, 'r') as file:
        data_config = yaml.safe_load(file)
    
    print(f"데이터 설정 정보: {data_config}")
    print(f"클래스 수: {data_config['nc']}")
    print(f"클래스 이름: {data_config['names']}")
    
    # CUDA 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # YOLOv11 모델 로드
    model_name = "yolo11n.pt"
    if not os.path.exists(model_name):
        print(f"오류: 모델 파일이 존재하지 않습니다: {model_name}")
        return None
        
    print(f"모델 파일을 로드합니다: {model_name}")
    model = YOLO(model_name)
    
    # 모델 학습
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0 if torch.cuda.is_available() else 'cpu',
        project="yolov11_smoke_detection",
        name="train_results",
        save=True,
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pretrained=True,
        verbose=True,
        seed=42,
        exist_ok=True
    )
    
    # 학습된 모델 경로
    trained_model_path = results.best
    print(f"최적 모델 저장 경로: {trained_model_path}")
    
    # 모델 검증
    metrics = model.val()
    print(f"검증 메트릭스: {metrics}")
    
    # 모델 예측 (테스트 이미지에 대해)
    # test_image_path = "test/images/example.jpg"  # 테스트할 이미지 경로
    # if os.path.exists(test_image_path):
    #     results = model.predict(test_image_path, save=True, conf=0.25)
    #     print(f"예측 결과: {results}")
    
    return trained_model_path

if __name__ == "__main__":
    print("YOLOv11 모델 학습 시작...")
    model_path = train_yolov11()
    if model_path:
        print(f"학습 완료! 모델 저장 경로: {model_path}")
    else:
        print("모델 학습 실패!") 