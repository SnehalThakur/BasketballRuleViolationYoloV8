import os

from ultralytics import YOLO


model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

model.train(data=r'C:\Users\snehal\PycharmProjects\AI-Basketball-Rule-Violation-Detection\config.yml', epochs=60, imgsz=640)