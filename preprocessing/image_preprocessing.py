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
import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO 이미지 전처리')
    parser.add_argument('--input-dir', type=str, required=True, help='입력 데이터 디렉토리 (필수)')
    parser.add_argument('--output-dir', type=str, required=True, help='출력 데이터 디렉토리 (필수)')
    return parser.parse_args()

args = None
INPUT_IMAGES_DIR = None
INPUT_LABELS_DIR = None
OUTPUT_IMAGES_DIR = None
OUTPUT_LABELS_DIR = None
TARGET_SIZE = (640, 640)

def setup_paths():
    global INPUT_IMAGES_DIR, INPUT_LABELS_DIR, OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR

    INPUT_IMAGES_DIR = os.path.join(args.input_dir, "images")
    INPUT_LABELS_DIR = os.path.join(args.input_dir, "labels")
    OUTPUT_IMAGES_DIR = os.path.join(args.output_dir, "images")
    OUTPUT_LABELS_DIR = os.path.join(args.output_dir, "labels")

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"오류: 입력 이미지 디렉토리가 존재하지 않습니다: {INPUT_IMAGES_DIR}")
        sys.exit(1)
    if not os.path.exists(INPUT_LABELS_DIR):
        print(f"오류: 입력 레이블 디렉토리가 존재하지 않습니다: {INPUT_LABELS_DIR}")
        sys.exit(1)

def process_image(image_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None, None

    original_height, original_width = image.shape[:2]
    ratio = min(TARGET_SIZE[0] / original_width, TARGET_SIZE[1] / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    padded_image = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.uint8)

    x_offset = (TARGET_SIZE[0] - new_width) // 2
    y_offset = (TARGET_SIZE[1] - new_height) // 2
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    cv2.imwrite(str(output_path), padded_image)

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
    transformed_points = []

    for i in range(0, len(segmentation), 2):
        if i + 1 < len(segmentation):
            x = segmentation[i]
            y = segmentation[i + 1]

            new_x = (x * transform_info['scale_x'] + transform_info['x_offset']) / TARGET_SIZE[0]
            new_y = (y * transform_info['scale_y'] + transform_info['y_offset']) / TARGET_SIZE[1]

            new_x = max(0, min(1, new_x))
            new_y = max(0, min(1, new_y))

            transformed_points.extend([new_x, new_y])

    return transformed_points

def process_json_to_yolo(json_path, transform_info):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    yolo_data = []
    # COCO category id → YOLO class id 맵핑
    category_map = {cat['id']: idx for idx, cat in enumerate(data.get('categories', []))}

    for annotation in data.get('annotations', []):
        # segmentation 정보가 없거나 비어 있으면 스킵
        seg_list = annotation.get('segmentation', [])
        if not seg_list or not seg_list[0]:
            continue
        segmentation = seg_list[0]

        category_id = annotation.get('category_id')
        yolo_class_id = category_map.get(category_id, category_id - 1)

        transformed_coords = transform_coordinates(segmentation, transform_info)
        yolo_line = [yolo_class_id] + transformed_coords
        yolo_data.append(yolo_line)

    return yolo_data

def save_yolo_label(yolo_data, output_path):
    with open(output_path, 'w') as f:
        for line in yolo_data:
            formatted_line = f"{int(line[0])}"
            for coord in line[1:]:
                formatted_line += f" {coord:.6f}"
            f.write(formatted_line + '\n')

def main():
    global args
    args = parse_args()
    setup_paths()

    print(f"입력 디렉토리: {args.input_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"타겟 크기: {TARGET_SIZE} (고정)")

    image_files = list(Path(INPUT_IMAGES_DIR).glob('*.jpg')) + \
                  list(Path(INPUT_IMAGES_DIR).glob('*.jpeg')) + \
                  list(Path(INPUT_IMAGES_DIR).glob('*.png'))

    if not image_files:
        print(f"오류: 처리할 이미지를 찾을 수 없습니다: {INPUT_IMAGES_DIR}")
        sys.exit(1)

    total = len(image_files)

    for i, image_path in enumerate(image_files, 1):
        image_base_name = image_path.stem
        label_path = Path(INPUT_LABELS_DIR) / f"{image_base_name}.json"
        output_image_path = Path(OUTPUT_IMAGES_DIR) / f"{image_base_name}.jpg"
        output_label_path = Path(OUTPUT_LABELS_DIR) / f"{image_base_name}.txt"

        if not label_path.exists():
            continue

        _, transform_info = process_image(image_path, output_image_path)
        if transform_info:
            yolo_data = process_json_to_yolo(label_path, transform_info)
            save_yolo_label(yolo_data, output_label_path)

        progress = (i / total) * 100
        print(f"\r진행률: {progress:.2f}% ({i}/{total})", end='', flush=True)

    print("\n모든 이미지와 라벨 전처리 완료!")

if __name__ == "__main__":
    main()