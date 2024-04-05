from ultralytics import YOLO
import cv2
import math 

# start webcam
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# emotions
model = YOLO("./runs/detect/train7/weights/best.pt")

# object classes
labels = [
    'Anger',
    'Contempt',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

while True:
    success, img = cap.read()
    emo_results = model(img, stream=True)

    # coordinates
    for r in emo_results:
        boxes = r.boxes
        i = 0
        for box in boxes:
            i += 1
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            # print("Class name -->", labels[cls])

            # object details
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2            
            title = labels[cls] + " " + str(confidence)

            cv2.putText(img, title, [50,50 * i], font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()