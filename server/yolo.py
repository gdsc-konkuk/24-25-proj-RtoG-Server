import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# YOLO 모델 로드 (애플리케이션 시작 시 한 번 로드 권장)
model = YOLO('./best.pt') # 예시 모델 경로
print("YOLO model loaded.") # 로딩 확인 로그

# 최근 3개 프레임의 화재 관련 객체 감지 여부 저장 (True: 감지됨, False: 감지 안됨)
recent_detections = deque(maxlen=3)

# 화재 의심 신고 로직 (일단 비워둠)
def fire_alert():
    """
    화재 의심 상황 시 호출될 함수.
    추후 실제 신고 또는 알림 로직 구현 필요.
    """
    print("화재 의심! 연속 3프레임에서 객체 감지됨.") # 임시 로그
    pass # TODO: 실제 화재 신고 로직 구현

async def process_frame_with_yolo(frame: np.ndarray) -> np.ndarray:
    """
    주어진 NumPy 배열 형태의 프레임에 대해 YOLO 객체 검출을 수행하고,
    객체가 마킹된 프레임을 반환합니다. 연속 3프레임 화재 감지 시 fire_alert 호출.
    """
    # ** 실제 YOLO 모델 추론 코드 작성 **
    # verbose=False로 로그 최소화
    results = model.predict(frame, verbose=False)

    # 현재 프레임에서 클래스 0, 1, 2 객체 감지 여부
    current_frame_detected = False

    # 결과에서 바운딩 박스, 클래스, 신뢰도 등을 추출하여 프레임에 마킹
    marked_frame = frame.copy()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                # 클래스 0, 1, 2 감지 시 플래그 설정
                if cls in [0, 1, 2]:
                    current_frame_detected = True

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

    # 최근 감지 결과 업데이트
    recent_detections.append(current_frame_detected)

    # 연속 3프레임 감지 확인 및 알림
    if len(recent_detections) == 3 and all(recent_detections):
        fire_alert()

    await asyncio.sleep(0.001)

    return marked_frame