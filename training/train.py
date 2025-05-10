import os
import torch
from ultralytics import YOLO
import shutil

# Set training output directory (local to training dir)
train_output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(train_output_dir, exist_ok=True)

# Set app model directory
app_model_dir = os.path.join('app')
os.makedirs(app_model_dir, exist_ok=True)

# Load model
model = YOLO('yolo11n.pt')

# Train model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolo11n_custom',
    project=train_output_dir
)

# Save model locally first
local_model_path = os.path.join(train_output_dir, 'yolo11n_custom.pt')
model.save(local_model_path)

# Copy only the model file to app directory
app_model_path = os.path.join(app_model_dir, 'yolo11n.pt')
shutil.copy2(local_model_path, app_model_path) 