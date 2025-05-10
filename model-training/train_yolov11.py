#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import torch
from ultralytics import YOLO
from pprint import pprint

def train_yolov11():
    try:
        # 현재 작업 디렉토리 확인
        print("현재 작업 디렉토리:", os.getcwd())
        
        # data.yaml 파일 경로
        data_yaml_path = "yolo_custom.yaml"
        
        # data.yaml 파일 존재 여부 확인
        if not os.path.exists(data_yaml_path):
            print(f"오류: YAML 파일이 존재하지 않습니다: {data_yaml_path}")
            return None
        
        # data.yaml 파일 읽기
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as file:
                data_config = yaml.safe_load(file)
            
            # 필수 키 확인
            required_keys = ['nc', 'names', 'path']
            for key in required_keys:
                if key not in data_config:
                    print(f"오류: YAML 파일에 '{key}' 키가 없습니다.")
                    return None
            
            print(f"데이터 설정 정보: {data_config}")
            print(f"클래스 수: {data_config['nc']}")  # 6개 클래스
            print(f"클래스 이름: {data_config['names']}")  # 클래스 이름 출력
            print(f"데이터 경로: {data_config['path']}")  # ../processed
        
        except yaml.YAMLError as e:
            print(f"YAML 파일 파싱 오류: {e}")
            return None
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            return None
        
        # CUDA 사용 가능 여부 확인
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"사용 장치: {'CUDA' if device == 0 else 'CPU'}")
        
        # YOLOv11 모델 로드
        model_name = "yolo11n.pt"
        if not os.path.exists(model_name):
            print(f"오류: 모델 파일이 존재하지 않습니다: {model_name}")
            return None
            
        print(f"모델 파일을 로드합니다: {model_name}")
        model = YOLO(model_name)
        
        # epochs와 patience 설정 조정
        epochs = 33
        patience = min(20, epochs // 2)  # epochs의 절반 또는 최대 20으로 설정
        
        # 모델 학습
        try:
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                workers=4,
                device=device,
                project="yolov11_smoke_detection",
                name="train_results",
                save=True,
                patience=patience,  # 수정된 patience 값
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
            
            # 학습 결과에서 유용한 정보 추출 및 출력
            print("\n" + "="*50)
            print("학습 결과 요약:")
            print("="*50)
            
            # 최적 모델 경로
            trained_model_path = results.best_model_path
            print(f"최적 모델 저장 경로: {trained_model_path}")
            
            # 저장 디렉터리 정보
            print(f"학습 결과 저장 디렉터리: {results.save_dir}")
            
            # 학습 완료된 에폭 수
            if hasattr(results, 'epoch'):
                print(f"학습 완료된 에폭 수: {results.epoch}")
            elif hasattr(results, 'epochs_completed'):
                print(f"학습 완료된 에폭 수: {results.epochs_completed}")
            else:
                print(f"학습 완료된 에폭 수: {epochs}")
            
            # 학습 성능 지표 출력
            print("\n성능 지표:")
            if hasattr(results, 'metrics') and results.metrics is not None:
                metrics = results.metrics
                # Box 성능
                if hasattr(metrics, 'box'):
                    print(f"  mAP50-95 (Box): {metrics.box.map:.4f}")
                    print(f"  mAP50 (Box): {metrics.box.map50:.4f}")
                
                # 클래스별 성능
                if hasattr(metrics, 'box') and hasattr(metrics.box, 'maps'):
                    print("\n클래스별 성능 (mAP50-95):")
                    for i, class_map in enumerate(metrics.box.maps):
                        class_name = data_config['names'].get(i, f"클래스 {i}")
                        print(f"  {class_name}: {class_map:.4f}")
            else:
                print("  성능 지표를 가져올 수 없습니다.")
            
            # 학습 손실 정보
            print("\n학습 손실 정보:")
            final_epoch = epochs - 1
            if hasattr(results, 'ema') and results.ema is not None:
                print(f"  최종 EMA 모델 사용")
            
            # 모델 크기 정보
            try:
                model_info = {}
                if hasattr(model, 'model'):
                    model_info['parameters'] = sum(p.numel() for p in model.model.parameters())
                    model_info['gradients'] = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                    print("\n모델 정보:")
                    print(f"  매개변수 수: {model_info['parameters']:,}")
                    print(f"  학습 가능한 매개변수 수: {model_info['gradients']:,}")
            except Exception as e:
                print(f"  모델 정보 계산 중 오류: {e}")
            
            # GPU 메모리 사용량 (CUDA인 경우)
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
                    print(f"  최대 GPU 메모리 사용량: {gpu_memory:.2f} MB")
                    torch.cuda.reset_peak_memory_stats()
                except Exception as e:
                    print(f"  GPU 메모리 정보 계산 중 오류: {e}")
            
            print("="*50 + "\n")
            
            return trained_model_path
            
        except Exception as e:
            print(f"모델 학습 중 오류 발생: {e}")
            return None
    
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        return None

if __name__ == "__main__":
    print("YOLOv11 모델 학습 시작...")
    model_path = train_yolov11()
    if model_path:
        print(f"학습 완료! 모델 저장 경로: {model_path}")
    else:
        print("모델 학습 실패!") 