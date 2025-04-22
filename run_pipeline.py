import os
import cv2
from glob import glob
from pathlib import Path
from ultralytics import YOLO
import shutil
import random
from concurrent.futures import ThreadPoolExecutor

# 경로 설정
VIDEO_DIR = "videos"
SEGMENT_DIR = "output/segments"
THUMBNAIL_DIR = "output/thumbnails"
ALERT_DIR = "output/alerts"
os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)
os.makedirs(ALERT_DIR, exist_ok=True)

# 영상 분할
def split_video(video_path, segment_length=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_name = Path(video_path).stem
    segments = []

    for i in range(0, total_frames, int(fps * segment_length)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        out_path = os.path.join(SEGMENT_DIR, f"{base_name}_part{i}.mp4")
        out = cv2.VideoWriter(out_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              (int(cap.get(3)), int(cap.get(4))))
        for _ in range(int(fps * segment_length)):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        segments.append(out_path)
    cap.release()
    return segments

# 썸네일 추출
def extract_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    thumb_path = os.path.join(THUMBNAIL_DIR, Path(video_path).stem + ".jpg")
    if ret:
        cv2.imwrite(thumb_path, frame)
    cap.release()
    return thumb_path

# 의심 판단
def is_suspected_fire(img_path, model):
    results = model(img_path)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in [0, 1, 2]:  # 흑색연기, 백색/회색연기, 화염
            return True
    return False


# mock이고, gemini call 해야함. 
def call_gemini_api_mock(img_path):
    return "화재 의심" if random.random() > 0.8 else "정상"

# 세그먼트 처리 (병렬용)
def process_segment(seg_path, model):
    thumb_path = extract_thumbnail(seg_path)
    if is_suspected_fire(thumb_path, model):
        decision = call_gemini_api_mock(thumb_path)
        if decision == "화재 의심":
            shutil.copy2(seg_path, ALERT_DIR)
            print(f"[🔥 ALERT] {Path(seg_path).name} 저장 완료")
        else:
            os.remove(thumb_path)  # 의심 아님 → 썸네일 삭제
    else:
        os.remove(thumb_path)  # YOLO 의심 안됨 → 썸네일 삭제

# 전체 실행
def process_all_videos_parallel(model_path="best.pt", max_workers=4):
    model = YOLO(model_path)
    video_files = glob(os.path.join(VIDEO_DIR, "*.mp4"))
    all_segments = []

    for video_path in video_files:
        segments = split_video(video_path)
        all_segments.extend(segments)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for seg_path in all_segments:
            executor.submit(process_segment, seg_path, model)

# 실행
if __name__ == "__main__":
    process_all_videos_parallel()
