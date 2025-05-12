## FastAPI 기반 실시간 객체 감지 영상 스트리밍 서버 구축 (프레임 단위 스트리밍)

### 기능 요구사항

- 서버에 저장된 영상을 읽어 프레임 단위로 YOLO 검증을 수행합니다.
- 검증과 동시에 객체가 마킹된 각 프레임을 클라이언트에게 실시간으로 스트리밍합니다.

### 구현 요구사항

- FastAPI 프레임워크를 사용하여 Python 서버를 구축합니다.
- 기존에 구현된 YOLO 검증 로직을 활용하되, 프레임 단위 처리를 지원하도록 수정합니다.

### 구현 단계

./server/routers/lives.py 에 작성

**3. 프레임 생성 및 YOLO 검증 후 스트리밍 함수 정의**

다음 코드를 바탕으로 재작성 (./server/yolo.py)

```python
async def frame_generator_with_yolo(video_path: str):
    """
    주어진 영상 파일 경로를 통해 프레임을 읽고, YOLO 검증 후 마킹된 프레임을 생성하여 yield 합니다.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=404, detail="영상을 열 수 없습니다.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 영상의 끝

            # YOLO 검증 및 프레임 마킹 (your_yolo_module의 비동기 함수 사용)
            marked_frame = await process_frame_with_yolo(frame)

            # 마킹된 프레임을 JPEG 형식으로 인코딩하여 스트리밍 형식에 맞게 yield
            _, encoded_frame = cv2.imencode('.jpg', marked_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')

        cap.release()

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프레임 처리 또는 스트리밍 중 오류 발생: {e}")
```

단순 검증 예시는 ./model-training/predict.py 에 정의되어있음.

**4. 실시간 YOLO 검증 및 스트리밍 엔드포인트 정의**

```python
@app.get("/stream_realtime/{video_name}")
async def stream_video_realtime(video_name: str):
    """
    특정 영상을 실시간으로 읽어 YOLO 검증을 수행하고, 마킹된 프레임을 스트리밍합니다.
    """
    video_path = f"{VIDEO_STORAGE_PATH}/{video_name}"
    return StreamingResponse(frame_generator_with_yolo(video_path), media_type="multipart/x-mixed-replace; boundary=frame")
```

**5. YOLO 검증 로직 (your_yolo_module.py - 수정)**

```python
# your_yolo_module.py
import asyncio
import cv2
import numpy as np
# 실제 YOLO 관련 라이브러리 및 모델 import 필요

# YOLO 모델 로드 (서버 시작 시 한 번만 로드하는 것을 권장)
# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')

async def process_frame_with_yolo(frame: np.ndarray) -> np.ndarray:
    """
    주어진 NumPy 배열 형태의 프레임에 대해 YOLO 객체 검출을 수행하고, 객체가 마킹된 프레임을 반환합니다.
    (실제 YOLO 모델 추론 및 프레임 마킹 로직 구현 필요)
    """
    # ** 실제 YOLO 모델 추론 코드 작성 **
    # 예시:
    # results = model.predict(frame)
    # 마킹된 프레임 생성 (bounding boxes, labels 등 그리기)
    marked_frame = frame.copy() # 임시로 원본 프레임 복사
    # 여기에 객체 마킹 로직 추가 (예: cv2.rectangle, cv2.putText)
    await asyncio.sleep(0.03) # 프레임 처리 시간 시뮬레이션
    return marked_frame
```

**6. 서버 실행**

```bash
uvicorn main:app --reload
```

### 클라이언트 구현 (예시 - 웹 브라우저 `<img src>` 태그 사용)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Real-time YOLO Stream</title>
</head>
<body>
    <h1>Real-time YOLO Stream</h1>
    <img src="http://localhost:8000/stream_realtime/your_video.mp4" />
</body>
</html>
```

### 주요 변경 사항 및 고려 사항

- **실시간 처리:** 서버는 영상을 **프레임 단위**로 처리하며, YOLO 검증과 마킹을 각 프레임에 대해 수행합니다.
- **스트리밍 방식:** `multipart/x-mixed-replace` 미디어 타입을 사용하여 서버에서 지속적으로 업데이트되는 JPEG 이미지를 클라이언트에게 전송합니다.
- **YOLO 로직 수정:** `your_yolo_module.py`의 `process_frame_with_yolo` 함수는 이제 **프레임(NumPy 배열)**을 입력으로 받아 마킹된 프레임을 반환해야 합니다.
- **클라이언트 구현:** 클라이언트는 이미지 스트림을 지원하는 방식으로 서버 엔드포인트에 연결해야 합니다. 웹 브라우저의 `<img src>` 태그가 간단한 예시입니다.
- **성능:** 실시간 성능은 YOLO 모델의 추론 속도, 프레임 처리 로직 효율성, 네트워크 환경 등에 따라 달라질 수 있습니다. 모델 최적화 및 효율적인 코딩이 중요합니다.
- **자원 관리:** 영상 파일을 읽고 프레임을 처리하는 과정에서 메모리 사용량이 증가할 수 있으므로, 적절한 자원 관리가 필요합니다.
- **오류 처리:** 영상 파일 오류, YOLO 처리 오류 등 발생 가능한 예외 상황에 대한 robust한 오류 처리가 중요합니다.

이 구조를 통해 서버는 영상을 읽는 즉시 각 프레임에 대해 YOLO 검증을 수행하고 결과를 클라이언트에게 실시간으로 스트리밍할 수 있습니다. 클라이언트 측에서는 해당 스트림을 이미지 형태로 표시하여 실시간 객체 감지 영상을 확인할 수 있습니다.