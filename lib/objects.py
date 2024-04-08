from ultralytics import YOLO
from lib.emotions import predict_emotion
import cv2
import math 

#objects
obj_model = YOLO("./yolov8n.pt")

obj_classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# object details
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2


def detect_objects(img, stream=False):
    obj_results = obj_model(img, stream)
    # coordinates
    for r in obj_results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values


            # class name
            cls = int(box.cls[0])
            # print("Class name -->", obj_classes[cls])
            
            caption = obj_classes[cls]
            
            # emotion

            if obj_classes[cls] == "person":
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)

                crop_img = img[y1:y2, x1:x2]
                emo, conf = predict_emotion(crop_img)
                caption = emo + " (" + str(confidence) + ")"
                cv2.putText(img, caption, [x1, y1], font, fontScale, color, thickness)
                print("Emotion -->", emo, conf)



            