import os
import json
import cv2
from glob import glob
from tqdm import tqdm
import numpy as np
import unicodedata

# ──────────────────────── #
# 설정
BASE_DIR = "Sample"
IMG_DIR = os.path.join(BASE_DIR, "01.원천데이터")
LBL_DIR = os.path.join(BASE_DIR, "02.라벨링데이터")

OUTPUT_IMG = "processed/images"
OUTPUT_LBL = "processed/labels"
os.makedirs(OUTPUT_IMG, exist_ok=True)
os.makedirs(OUTPUT_LBL, exist_ok=True)

# ──────────────────────── #
# Letterbox (YOLO-style resize)
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

# ──────────────────────── #
# 유니코드 정규화
def normalize_filename(name):
    return unicodedata.normalize("NFC", name)

def build_image_path_map(image_root):
    image_map = {}
    for root, _, files in os.walk(image_root):
        for fname in files:
            if fname.lower().endswith(".jpg"):
                norm_name = normalize_filename(fname)
                image_map[norm_name] = os.path.join(root, fname)
    return image_map

def find_image_path(image_map, file_name):
    norm_name = normalize_filename(file_name)
    return image_map.get(norm_name)

# ──────────────────────── #
# COCO JSON 처리
def process_json(json_path, image_map):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    id_to_info = {img["id"]: img for img in data["images"]}
    cat_map = {cat["id"]: idx for idx, cat in enumerate(data["categories"])}
    labels = {}

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        img_info = id_to_info[img_id]
        file_name = img_info["file_name"]
        file_id = os.path.splitext(file_name)[0]

        img_path = find_image_path(image_map, file_name)
        if not img_path:
            print(f"[WARN] 이미지 못 찾음: {file_name}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 이미지 로드 실패: {img_path}")
            continue

        resized = letterbox(img)
        save_path = os.path.join(OUTPUT_IMG, f"{file_id}.jpg")
        cv2.imwrite(save_path, resized)

        x, y, w, h = ann["bbox"]
        x_c = (x + w / 2) / img_info["width"]
        y_c = (y + h / 2) / img_info["height"]
        w_n = w / img_info["width"]
        h_n = h / img_info["height"]
        class_id = cat_map[ann["category_id"]]
        line = f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
        labels.setdefault(file_id, []).append(line)

    for fid, lines in labels.items():
        out_path = os.path.join(OUTPUT_LBL, f"{fid}.txt")
        with open(out_path, 'w') as f:
            f.write("\n".join(lines))

# ──────────────────────── #
# 실행
if __name__ == "__main__":
    print("📦 이미지 경로 맵핑 중...")
    image_map = build_image_path_map(IMG_DIR)

    print("🔎 JSON 전처리 시작...")
    json_files = glob(os.path.join(LBL_DIR, "**/*.json"), recursive=True)
    for jf in tqdm(json_files, desc="🚧 전처리 중"):
        process_json(jf, image_map)

    print("✅ 전처리 완료: processed/images, processed/labels 에 저장됨")