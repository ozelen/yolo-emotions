from ultralytics import YOLO
from lib.emotions import predict_emotion
from lib.objects import detect_objects

import cv2
import math 

# start webcam
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    obj_results = detect_objects(img, stream=True)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()