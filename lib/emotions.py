from ultralytics import YOLO
import cv2
import math 

# emotions
emo_model = YOLO("./runs/detect/train7/weights/best.pt")

# emotion classes
emo_classes = [
    'Anger',
    'Contempt',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

def predict_emotion(img, stream=False):
    emo_results = emo_model(img, stream)

    for r in emo_results:
        boxes = r.boxes
        for box in boxes:
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            # print("Class name -->", labels[cls])
            return emo_classes[cls], confidence

    return "No Emotion", 0
