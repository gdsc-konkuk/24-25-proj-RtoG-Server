import os
import cv2
import numpy as np
from typing import List, Tuple
from ..core.config import settings

class VideoService:
    def __init__(self):
        self.upload_dir = settings.VIDEO_UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)

    def process_video(self, video_path: str) -> List[Tuple[float, float, float]]:
        """Process video and return list of (timestamp, x, y) coordinates."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        coordinates = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    timestamp = frame_count / fps
                    coordinates.append((timestamp, cx, cy))

            frame_count += 1

        cap.release()
        return coordinates

    def save_video(self, file) -> str:
        """Save uploaded video file and return the path."""
        file_path = os.path.join(self.upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)
        return file_path

video_service = VideoService() 