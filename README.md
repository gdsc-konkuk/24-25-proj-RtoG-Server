# 🔥 RtoG: Wildfire Fast Detection Server (Red to Green)

This repository powers the RtoG (Red to Green) server — an AI-driven system for fast and accurate wildfire detection.
It uses YOLOv8 for real-time object detection and integrates Gemini Vision API for verification, enabling early fire detection from video footage like CCTV or drone sources.

From red (wildfire) to green (safe forest) — detect fast, act faster.

---

## 📁 Project Structure

```
RtoG/
├── preprocessing/         # Image preprocessing tools
│   ├── Dockerfile        # Docker environment for preprocessing
│   └── image_preprocessing.py # COCO to YOLO format converter
├── processed/             # Resized images + YOLO-format labels
├── runs/                  # Training results (✅ YOLOv8 output, only `best.pt` included)
├── Sample/                # Original dataset  
│   ├── 01.원천데이터/         # Images in nested folders (JPG)  
│   ├── 02.라벨링데이터/        # COCO-style JSON labels  
│   ├── data/              # Copied source data
│   ├── processed/         # Processed images and labels
│   └── result/            # Visualization results
│   ❌ Not included in repo (COCO-style dataset, add manually)
├── videos/                # Raw CCTV or simulated wildfire videos (MP4)  
│   ❌ Not included in repo (CCTV/wildfire videos, add manually)
├── yolov8env310/          # Python virtual environment (optional)
├── check_images.py        # Segmentation visualization tool
├── process_and_detect.py  # Converts COCO to YOLO format (with letterbox resize)
├── relocate_sample_dir.py # Sample directory management tool
├── run_pipeline.py        # 10s video segmentation → YOLO → Gemini → Alert
├── yolo_custom.yaml       # Dataset config file for YOLOv8 training
└── yolov8n.pt             # Pretrained YOLOv8 base model (for transfer learning)
```

---

## 1. Dataset Processing
### 1.1 Relocate Sample Dataset
```bash
python relocate_sample_dir.py
```
- Organizes original dataset in structured directories
- Copies source images to `Sample/data/images`
- Copies label JSONs to `Sample/data/labels`

### 1.2 Preprocess COCO Dataset to YOLO Format
```bash
python process_and_detect.py
```
- Walks through nested folders
- Matches JSON labels to original JPGs
- Resizes images to 640x640 (letterbox style)
- Converts bounding boxes to YOLO `.txt` format

### 1.3 Segmentation Visualization
```bash
python check_images.py
```
- Visualizes YOLO-format segmentation labels
- Creates overlay with class colors and labels
- Saves visualization results to `Sample/result`

---

## 2. Train YOLOv8 Model
```bash
yolo detect train \
    data=yolo_custom.yaml \
    model=yolov8n.pt \
    epochs=50 \
    imgsz=640 \
    batch=16 \
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
- `ultralytics`, `opencv-python`, `tqdm`, `numpy`, `pillow`

```bash
pip install ultralytics opencv-python tqdm numpy pillow
```

---

## 📍 Notes
- Original image size is preserved during letterbox resize
- `runs/` folder may grow large, clean periodically
- Sample/ and videos/ are not included in the repository due to size constraints.   
➤ Please add them manually to the root directory if you want to run the full pipeline.
- Only best.pt from runs/ is included to keep the repo size reasonable.

