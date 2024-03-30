from PIL import Image
import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from ultralytics import YOLO
import double_dribble_detector as dd

detector = dd.DoubleDribbleDetector()
p_time = 0

st.title("Basketball Rule Violation Detection")
sample_img = cv2.imread('referee.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
frameplaceholder = st.empty()
cap = None

# YOLOv8 Model
model_type = "YoloV8-Pose"
model = YOLO("yolov8n-pose-best.pt")


# Inference Mode
options = st.sidebar.radio(
    'Options:', ('Webcam', 'Video'), index=1)


# Video
if options == 'Video':
    upload_video_file = st.sidebar.file_uploader(
        'Upload Video', type=['mp4', 'avi', 'mkv'])
    if upload_video_file is not None:
        pred = st.checkbox(f'Predict Using {model_type}')

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload_video_file.read())
        cap = cv2.VideoCapture(tfile.name)

# Webcam
if options == 'Webcam':
    cam_options = st.sidebar.selectbox('Webcam Channel',
                                       ('Select Channel', '0', '1', '2', '3'))

    if not cam_options == 'Select Channel':
        pred = st.checkbox(f'Predict Using {model_type}')
        cap = cv2.VideoCapture(int(cam_options))

if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    while True:
        success, frame = cap.read()
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!", icon="🚨"
            )
            break

        if success:
            pose_annotated_frame, ball_detected = detector.process_frame(frame)
            detector.check_double_dribble()
            if detector.double_dribble_time and time.time() - detector.double_dribble_time <= 3:
                red_tint = np.full_like(pose_annotated_frame, (0, 0, 255), dtype=np.uint8)
                pose_annotated_frame = cv2.addWeighted(pose_annotated_frame, 0.7, red_tint, 0.3, 0)
                cv2.putText(pose_annotated_frame, "Double dribble detected!", (detector.frame_width - 600, 150,),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA, )

        stframe1.image(pose_annotated_frame, channels="RGB")
    cap.release()
