"""
Image Preprocessing for YOLO 11 Training
----------------------------------------
This script preprocesses images and their corresponding segmentation labels from COCO JSON format
to YOLO compatible format while maintaining the correct aspect ratio and segmentation coordinates.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

# 경로 설정
INPUT_IMAGES_DIR = "data/images"
INPUT_LABELS_DIR = "data/labels"
OUTPUT_IMAGES_DIR = "processed/images"
OUTPUT_LABELS_DIR = "processed/labels"
TARGET_SIZE = (640, 640)  # YOLO v11 타겟 크기

# 출력 디렉토리 생성
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

def process_image(image_path, output_path):
    """이미지를 로드하고 타겟 크기로 리사이즈 (종횡비 유지)"""
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None, None
    
    original_height, original_width = image.shape[:2]
    
    # 종횡비 계산 및 새 크기 결정
    ratio = min(TARGET_SIZE[0] / original_width, TARGET_SIZE[1] / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    # 이미지 리사이즈
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 패딩 추가하여 640x640 만들기
    padded_image = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)
    
    # 이미지를 중앙에 배치
    x_offset = (TARGET_SIZE[0] - new_width) // 2
    y_offset = (TARGET_SIZE[1] - new_height) // 2
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    
    # 이미지 저장
    cv2.imwrite(str(output_path), padded_image)
    
    # 변환에 필요한 정보 반환
    transform_info = {
        'original_width': original_width,
        'original_height': original_height,
        'new_width': new_width,
        'new_height': new_height,
        'x_offset': x_offset,
        'y_offset': y_offset,
        'scale_x': ratio,
        'scale_y': ratio
    }
    
    return padded_image, transform_info

def transform_coordinates(segmentation, transform_info):
    """세그먼테이션 좌표를 변환"""
    transformed_points = []
    
    for i in range(0, len(segmentation), 2):
        if i + 1 < len(segmentation):
            x = segmentation[i]
            y = segmentation[i + 1]
            
            # 원본 이미지에서의 비율 계산
            x_ratio = x / transform_info['original_width']
            y_ratio = y / transform_info['original_height']
            
            # 새 이미지에서의 좌표 계산 (스케일링 + 오프셋)
            new_x = (x * transform_info['scale_x'] + transform_info['x_offset']) / TARGET_SIZE[0]
            new_y = (y * transform_info['scale_y'] + transform_info['y_offset']) / TARGET_SIZE[1]
            
            # 유효한 범위(0~1) 내로 제한
            new_x = max(0, min(1, new_x))
            new_y = max(0, min(1, new_y))
            
            transformed_points.extend([new_x, new_y])
    
    return transformed_points

def process_json_to_yolo(json_path, transform_info):
    """COCO JSON 형식을 YOLO 형식으로 변환"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    yolo_data = []
    
    # 카테고리 ID 매핑 (COCO는 1부터 시작, YOLO는 0부터 시작)
    category_map = {}
    for idx, category in enumerate(data.get('categories', [])):
        category_map[category['id']] = idx
    
    # 어노테이션 처리
    for annotation in data.get('annotations', []):
        category_id = annotation.get('category_id')
        segmentation = annotation.get('segmentation', [[]])[0]  # 첫 번째 세그먼테이션 다각형 사용
        
        # YOLO 형식으로 클래스 ID 변환 (0부터 시작)
        yolo_class_id = category_map.get(category_id, category_id - 1)
        
        # 좌표 변환
        transformed_coords = transform_coordinates(segmentation, transform_info)
        
        # YOLO 형식: class_id x1 y1 x2 y2 ... xn yn
        yolo_line = [yolo_class_id] + transformed_coords
        yolo_data.append(yolo_line)
    
    return yolo_data

def save_yolo_label(yolo_data, output_path):
    """YOLO 형식의 라벨을 텍스트 파일로 저장"""
    with open(output_path, 'w') as f:
        for line in yolo_data:
            # 클래스 ID를 정수로, 좌표를 소수점으로 포맷팅
            formatted_line = f"{int(line[0])}"
            for coord in line[1:]:
                formatted_line += f" {coord:.6f}"
            f.write(formatted_line + '\n')

def main():
    """메인 처리 함수"""
    # 이미지 디렉토리에서 모든 이미지를 처리
    image_files = list(Path(INPUT_IMAGES_DIR).glob('*.jpg')) + list(Path(INPUT_IMAGES_DIR).glob('*.jpeg')) + list(Path(INPUT_IMAGES_DIR).glob('*.png'))
    
    for image_path in image_files:
        # 이미지 기본 이름 얻기
        image_base_name = image_path.stem
        
        # 해당 라벨 파일 찾기
        label_path = Path(INPUT_LABELS_DIR) / f"{image_base_name}.json"
        
        if not label_path.exists():
            print(f"라벨 파일을 찾을 수 없습니다: {label_path}")
            continue
        
        # 출력 파일 경로 설정
        output_image_path = Path(OUTPUT_IMAGES_DIR) / f"{image_base_name}.jpg"
        output_label_path = Path(OUTPUT_LABELS_DIR) / f"{image_base_name}.txt"
        
        print(f"처리 중: {image_path.name}")
        
        # 이미지 처리
        _, transform_info = process_image(image_path, output_image_path)
        
        if transform_info:
            # JSON에서 YOLO 형식으로 변환
            yolo_data = process_json_to_yolo(label_path, transform_info)
            
            # YOLO 라벨 저장
            save_yolo_label(yolo_data, output_label_path)
            
            print(f"성공적으로 변환 완료: {output_image_path.name}, {output_label_path.name}")

if __name__ == "__main__":
    main()
    print("모든 이미지와 라벨 전처리 완료!")