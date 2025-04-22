# 🔥 RtoG: Wildfire Detection Pipeline (YOLOv8 + Gemini Vision API)

This repository contains a wildfire detection pipeline using the Ultralytics YOLOv8 object detection model. It converts COCO-format fire datasets into YOLO format, trains the model, and uses video footage to detect early signs of wildfire.

---

## 📁 Project Structure

```
RtoG/
├── processed/            # Resized images + YOLO-format labels
├── runs/                 # Training results (✅ YOLOv8 output, only best.pt included)
├── Sample/               # Original dataset (❌ Not included in repo (COCO-style dataset, add manually))
│   ├── 01.원천데이터/        # Images in nested folders (JPG)
│   └── 02.라벨링데이터/       # COCO-style JSON labels
├── videos/               # Raw CCTV or simulated wildfire videos (MP4) (❌ Not included in repo (CCTV/wildfire videos, add manually))
├── yolov8env310/         # Python virtual environment (optional)
├── process_and_detect.py # Converts COCO to YOLO format (with letterbox resize)
├── run_pipeline.py       # 10s video segmentation → YOLO → Gemini → Alert
├── yolo_custom.yaml      # Dataset config file for training
└── yolov8n.pt            # Base YOLOv8 model (pretrained)
```

---

## 1. Preprocess COCO Dataset to YOLO Format
```bash
python process_and_detect.py
```
- Walks through nested folders
- Matches JSON labels to original JPGs
- Resizes images to 640x640 (letterbox style)
- Converts bounding boxes to YOLO `.txt` format

---

## 2. Train YOLOv8 Model
```bash
yolo detect train \
    data=yolo_custom.yaml \
    model=yolov8n.pt \
    epochs=12 \
    imgsz=416 \
    batch=4 \
    project=runs \
    name=train
```
- Outputs to: `runs/train/weights/best.pt`
- You can change `imgsz`, `epochs`, etc.

---

## 3. Run Inference on Videos (10s segmentation)
```bash
python run_pipeline.py
```
- Segments videos into 10-second clips
- Extracts thumbnail from each segment
- Runs YOLO on thumbnail using trained `best.pt`
- If suspicious → calls Gemini Vision API
- If fire is confirmed → stores video block & thumbnail

---

## 🧠 Model Class Mapping
| ID | Class Label       |
|----|-------------------|
| 0  | 흑색연기 (Black Smoke)   |
| 1  | 백색/회색연기 (White/Grey Smoke) |
| 2  | 화염 (Flame)          |
| 3  | 구름 (Cloud)         |
| 4  | 안개/연무 (Fog/Mist)    |
| 5  | 굴뚝연기 (Chimney Smoke) |

---

## 🔌 Gemini API Integration
- Current: `call_gemini_api_mock()` (placeholder)
- Replace with actual Gemini Vision API call logic for production

---

## 📦 Requirements
- Python 3.10+
- `ultralytics`, `opencv-python`, `tqdm`, `numpy`.

```bash
pip install ultralytics opencv-python tqdm numpy
```

---

## 📍 Notes
- Original image size is preserved during letterbox resize
- `runs/` folder may grow large, clean periodically
- You can stop/resume training via `resume=True`
- Sample/ and videos/ are not included in the repository due to size constraints.
➤ Please add them manually to the root directory if you want to run the full pipeline.
- Only best.pt from runs/ is included to keep the repo size reasonable.

