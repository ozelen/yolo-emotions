from ultralytics import YOLO
import cv2
import math 

# start webcam
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

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

def predict_emotion(img):
    emo_results = emo_model(img, stream=True)

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

while True:
    success, img = cap.read()
    obj_results = obj_model(img, stream=True)

    # coordinates
    for r in obj_results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", obj_classes[cls])
            
            caption = obj_classes[cls]
            
            # emotion

            if obj_classes[cls] == "person":
                crop_img = img[y1:y2, x1:x2]
                emo, conf = predict_emotion(crop_img)
                caption = caption + " " + emo + " " + str(confidence)

            print("Emotion -->", emo, conf)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, caption, org, font, fontScale, color, thickness)


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()