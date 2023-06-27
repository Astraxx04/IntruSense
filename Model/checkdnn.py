import cv2
import numpy as np
import imutils
import os
import time

modelFile = "./face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./face_detection_model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
video_capture = cv2.VideoCapture(0)
sampleN = 0
frame_counter = 0
capture_interval = 40

people = []
DIR = r'dataset'
for person in os.listdir(DIR):
    people.append(person)
print(people)

start_time = time.time()
print("Collecting data...")

while True:
    ret, frame = video_capture.read()
 
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face = cv2.resize(frame, (300, 300))
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.8:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        sampleN += 1

        id = "gagan"
        try:
            path = os.path.join("dataset/", id)
            os.mkdir(path)
        except FileExistsError:
            print()
        finally:
            path = "Dataset/" + id + "/"
            cv2.imwrite(path + str(id) + str(sampleN) + ".jpg", face)
            sampleN += 1
            if sampleN > 199:
                exit()

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    print("Data collected...")
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    frame_counter += 1
    elapsed_time = time.time() - start_time

    if frame_counter >= capture_interval:
        if elapsed_time >= 1:
            frame_counter = 0
            start_time = time.time()

    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
video_capture.release()