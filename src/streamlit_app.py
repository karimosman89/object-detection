import streamlit as st
import cv2
from yolo import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
yolo = YOLO('yolov3.weights', 'yolov3.cfg', 'coco.names')

st.title("Object Detection with YOLO")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    detections = yolo.detect(image)
    
    # Draw bounding boxes
    for (box, class_id) in detections:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Show the result
    st.image(image, caption="Processed Image")

