from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m-pose.pt')  # load an official model
# model = YOLO('path/to/yolov8n-pose-best.pt')  # load a custom model

# Predict with the model
# results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

results = model(source=r"C:\Users\snehal\PycharmProjects\PoseEstimationTrainer\data\tadasana.png", show=True, conf=0.3, save=True)
