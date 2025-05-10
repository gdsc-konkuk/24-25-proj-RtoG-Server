# 데이터 전처리 모듈 (Preprocessing)

이 모듈은 화재 감지를 위한 이미지 데이터 전처리 작업을 담당합니다.

## 기능

- 이미지 전처리 및 정규화
- YOLO 형식 라벨 변환
- 데이터셋 준비

## 설치 및 실행

1. 가상환경 활성화:
```bash
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 전처리 실행:
```bash
python image_preprocessing.py --input-dir /경로/입력/디렉토리 --output-dir /경로/출력/디렉토리
```

## 디렉토리 구조

- `processed/`: 처리된 이미지와 라벨 파일이 저장되는 디렉토리

## 도커 컨테이너

이 도커 컨테이너는 YOLO 11 학습을 위한 이미지 전처리 작업을 수행합니다. 이미지와 COCO JSON 형식의 세그먼테이션 레이블을 입력받아 YOLO 호환 형식으로 변환합니다.

## 도커 컨테이너 실행 방법

### 간단한 실행 예제

```bash
# 입력 및 출력 디렉토리만 볼륨으로 마운트하면 됩니다
docker run --rm \
  -v /absolute/path/to/input:/app/data \
  -v /absolute/path/to/output:/app/processed \
  rtog-preprocessing
```

### GPU 활용 실행

```bash
# GPU 지원 활성화
docker run --rm --gpus all \
  -v /absolute/path/to/input:/app/data \
  -v /absolute/path/to/output:/app/processed \
  rtog-preprocessing
```

## 볼륨 마운트 설명

| 컨테이너 경로 | 용도 | 호스트 경로(예) |
|---------|------|--------|
| `/app/data` | 입력 데이터 디렉토리 | `/absolute/path/to/input` |
| `/app/processed` | 출력 데이터 디렉토리 | `/absolute/path/to/output` |

## 입출력 데이터 구조

### 입력 데이터 구조 (필수)

```
input_directory/
  ├── images/          # 원본 이미지 파일들 (.jpg, .jpeg, .png)
  └── labels/          # COCO JSON 형식의 레이블 파일들 (.json)
```

### 출력 데이터 구조

```
output_directory/
  ├── images/          # 처리된 이미지 파일들 (.jpg)
  └── labels/          # YOLO 형식의 레이블 파일들 (.txt)
```

## 기술 정보

- 이미지는 종횡비를 유지하면서 640x640 크기로 리사이징됩니다.
- 세그먼테이션 레이블이 YOLO 형식으로 변환됩니다.
- 처리된 모든 이미지는 JPG 형식으로 저장됩니다.

## 오류 처리

- 입력 디렉토리가 없거나 이미지가 없는 경우 오류가 발생하고 프로세스가 종료됩니다.
- 각 이미지에 대한 JSON 레이블 파일이 없는 경우, 해당 이미지는 건너뛰고 계속 진행합니다. 