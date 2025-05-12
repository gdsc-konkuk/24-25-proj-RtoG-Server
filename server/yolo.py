import asyncio
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드 (애플리케이션 시작 시 한 번 로드 권장)
model = YOLO('./best.pt') # 예시 모델 경로
print("YOLO model loaded.") # 로딩 확인 로그

async def process_frame_with_yolo(frame: np.ndarray) -> np.ndarray:
    """
    주어진 NumPy 배열 형태의 프레임에 대해 YOLO 객체 검출을 수행하고,
    객체가 마킹된 프레임을 반환합니다.
    """
    # ** 실제 YOLO 모델 추론 코드 작성 **
    # verbose=False로 로그 최소화
    results = model.predict(frame, verbose=False)

    # 결과에서 바운딩 박스, 클래스, 신뢰도 등을 추출하여 프레임에 마킹
    marked_frame = frame.copy()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                # model.names가 존재하고 cls가 유효한 인덱스인지 확인
                if hasattr(model, 'names') and cls < len(model.names):
                    label = f"{model.names[cls]} {conf:.2f}"
                else:
                    label = f"Class {cls} {conf:.2f}"

                # 바운딩 박스 그리기
                cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 라벨 텍스트 그리기
                cv2.putText(marked_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing box: {e}") # 개별 박스 처리 오류 로깅
                continue # 오류 발생 시 다음 박스로 넘어감

    await asyncio.sleep(0.001)

    return marked_frame