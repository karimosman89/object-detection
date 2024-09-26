import cv2
from yolo import YOLO

# Load YOLO
yolo = YOLO('yolov3.weights', 'yolov3.cfg', 'coco.names')

# Load video
cap = cv2.VideoCapture('data/test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = yolo.detect(frame)

    for (box, class_id) in detections:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
