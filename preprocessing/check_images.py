"""
세그먼테이션 시각화 도구
-----------------------
이 스크립트는 처리된 이미지와 YOLO 형식의 세그먼테이션 라벨이 올바르게 매칭되는지 시각적으로 확인합니다.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont

# 경로 설정
PROCESSED_IMAGES_DIR = "./Sample/processed/images"
PROCESSED_LABELS_DIR = "./Sample/processed/labels"

# 클래스 이름 정의 (YAML 파일과 동일하게 설정)
CLASS_NAMES = {
    0: "흑색연기",
    1: "백색/회색연기",
    2: "화염", 
    3: "구름",
    4: "안개/연무",
    5: "굴뚝연기"
}

# 각 클래스별 색상 랜덤 생성 (RGB 형식)
def generate_colors(n_classes):
    colors = {}
    for i in range(n_classes):
        # 무작위 색상 생성 (파스텔 색상을 피하기 위해 120-250 범위 사용)
        color = (
            random.randint(120, 250),
            random.randint(120, 250),
            random.randint(120, 250)
        )
        colors[i] = color
    return colors

# YOLO 형식 세그먼테이션 라벨 파싱
def parse_yolo_segmentation(label_path):
    segments = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:  # 최소 class_id + 2개의 점 (x,y) 필요
                continue
                
            class_id = int(parts[0])
            coords = [float(coord) for coord in parts[1:]]
            
            # x,y 쌍으로 변환
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append((coords[i], coords[i+1]))
            
            segments.append({
                'class_id': class_id,
                'points': points
            })
    return segments

# 세그먼테이션 시각화 (PIL 사용)
def visualize_segmentation(image_path, label_path, colors):
    # OpenCV로 이미지 로드
    cv_image = cv2.imread(str(image_path))
    if cv_image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None
    
    # OpenCV BGR -> RGB 변환
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # PIL 이미지로 변환
    pil_image = Image.fromarray(rgb_image)
    
    # 이미지 크기
    width, height = pil_image.size
    
    # 라벨 로드 및 파싱
    segments = parse_yolo_segmentation(label_path)
    
    # 세그먼테이션 마스크 생성 (투명 레이어)
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # 메인 이미지에 그리기
    draw = ImageDraw.Draw(pil_image)
    
    # 시스템 폰트 로드 (한글 지원)
    try:
        # macOS 기본 폰트
        font = ImageFont.truetype("AppleGothic.ttf", 15)
    except IOError:
        try:
            # Windows 기본 폰트
            font = ImageFont.truetype("malgun.ttf", 15)
        except IOError:
            try:
                # Linux 기본 폰트
                font = ImageFont.truetype("NanumGothic.ttf", 15)
            except IOError:
                # 폰트를 찾을 수 없으면 기본 폰트 사용
                font = ImageFont.load_default()
                print("경고: 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
    
    # 각 세그먼테이션 폴리곤 그리기
    for segment in segments:
        class_id = segment['class_id']
        points = segment['points']
        
        # 정규화된 좌표를 실제 픽셀 좌표로 변환
        pixel_points = []
        for x, y in points:
            px = int(x * width)
            py = int(y * height)
            pixel_points.append((px, py))
        
        # 색상 선택
        color = colors.get(class_id, (255, 0, 0))
        
        # 폴리곤 외곽선 그리기
        draw.polygon(pixel_points, outline=color, fill=None, width=2)
        
        # 반투명 채우기
        fill_color = color + (100,)  # 알파 채널 추가 (투명도)
        draw_overlay.polygon(pixel_points, fill=fill_color)
        
        # 클래스 라벨 표시
        if len(points) > 0:
            # 폴리곤의 중심점 계산
            centroid_x = int(sum(p[0] for p in points) * width / len(points))
            centroid_y = int(sum(p[1] for p in points) * height / len(points))
            
            # 클래스 이름
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            
            # 텍스트 크기 측정
            text_bbox = draw.textbbox((0, 0), class_name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 텍스트 배경 그리기
            draw.rectangle(
                [
                    centroid_x - 5, 
                    centroid_y - text_height - 5, 
                    centroid_x + text_width + 5, 
                    centroid_y + 5
                ], 
                fill=(0, 0, 0)
            )
            
            # 텍스트 그리기
            draw.text(
                (centroid_x, centroid_y - text_height),
                class_name,
                font=font,
                fill=(255, 255, 255)
            )
    
    # 오버레이 합성
    pil_image = pil_image.convert('RGBA')
    result = Image.alpha_composite(pil_image, overlay)
    result = result.convert('RGB')
    
    # PIL -> OpenCV 변환 (다시 BGR로)
    result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    
    return result_cv

def main():
    colors = generate_colors(len(CLASS_NAMES))
    
    # 처리된 이미지 디렉토리에서 모든 이미지를 확인
    image_files = list(Path(PROCESSED_IMAGES_DIR).glob("*.jpg")) + list(Path(PROCESSED_IMAGES_DIR).glob("*.jpeg")) + list(Path(PROCESSED_IMAGES_DIR).glob("*.png"))
    
    print(f"총 {len(image_files)}개의 이미지를 찾았습니다.")
    
    # 결과 저장 디렉토리
    output_dir = "Sample/result"
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path in image_files:
        image_base_name = image_path.stem
        label_path = Path(PROCESSED_LABELS_DIR) / f"{image_base_name}.txt"
        
        if not label_path.exists():
            print(f"라벨 파일을 찾을 수 없습니다: {label_path}")
            continue
        
        print(f"처리 중: {image_path.name}")
        
        # 세그먼테이션 시각화
        result_image = visualize_segmentation(image_path, label_path, colors)
        
        if result_image is not None:
            # 시각화 결과 저장
            output_path = os.path.join(output_dir, f"vis_{image_base_name}.jpg")
            cv2.imwrite(output_path, result_image)
            print(f"시각화 완료: {output_path}")
            
            # 결과 표시 제거 (사용자 요청에 따라)
    
    print("모든 이미지 시각화 완료!")

if __name__ == "__main__":
    main()
