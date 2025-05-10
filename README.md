# 🔥 RtoG: Wildfire Fast Detection Server (Red to Green)

This repository powers the RtoG (Red to Green) server — an AI-driven system for fast and accurate wildfire detection.
It uses YOLOv8 for real-time object detection and integrates Gemini Vision API for verification, enabling early fire detection from video footage like CCTV or drone sources.

From red (wildfire) to green (safe forest) — detect fast, act faster.

---

## 🚀 Project Structure (Mono Repository)

This project has been refactored into a monorepo with independent virtual environments:

```
RtoG/
├── api-server/           # Web server (FastAPI)
│   ├── venv/             # App server virtual environment
│   └── requirements.txt  # App server dependencies
│
├── preprocessing/        # Image preprocessing tools
│   ├── venv/             # Preprocessing virtual environment
│   └── requirements.txt  # Preprocessing dependencies
│
├── model-training/       # Model training code
│   ├── venv/             # Model training virtual environment
│   └── requirements.txt  # Model training dependencies
│
├── Other project files...
```

## 💻 Virtual Environment Setup

Each project uses an independent virtual environment to prevent dependency conflicts:

### api-server
```bash
cd api-server
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### preprocessing
```bash
cd preprocessing
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### model-training
```bash
cd model-training
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## 🧠 Model Class Mapping
| ID | Class Label       |
|----|-------------------|
| 0  | 흑색연기 (Black Smoke)   |
| 1  | 백색/회색연기 (White/Grey Smoke) |
| 2  | 화염 (Flame)          |
| 3  | 구름 (Cloud)         |
| 4  | 안개/연무 (Fog/Mist)    |
| 5  | 굴뚝연기 (Chimney Smoke) |
