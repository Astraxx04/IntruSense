import os
import cv2
import imutils
import time
import pickle
import numpy as np
import pandas as pd
from imutils.video import FPS
from imutils.video import VideoStream

# load serialized face detector
print("Loading Face Detector...")
protoPath = r"/Users/gagans/Desktop/face-recognition-using-deep-learning/face_detection_model/deploy.prototxt"
modelPath = r"/Users/gagans/Desktop/face-recognition-using-deep-learning/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch(r"/Users/gagans/Desktop/face-recognition-using-deep-learning/face_detection_model/openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(r"/Users/gagans/Desktop/face-recognition-using-deep-learning/output/recognizer", "rb").read())
le = pickle.loads(open(r"/Users/gagans/Desktop/face-recognition-using-deep-learning/output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

start_time = time.time()
person_identified = False

# DataFrame for identified persons
data = []

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # initialize flag for face presence in the current frame
    face_detected = False

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence >= 0.8:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            # print(proba)
            # if the probability is below a certain threshold, consider it as an unknown face
            if proba < 0.8:
                name = "Unknown"
            else:
                name = le.classes_[j]

            # print(name)
            # draw the bounding box of the face along with the associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # set the face_detected flag to True
            face_detected = True

            # check if enough time has elapsed since the last person identification
            elapsed_time = time.time() - start_time
            if elapsed_time >= 5.0 and not person_identified:
                # output the identified person
                print("Identified Person:", name)
                # append the identified person and the corresponding time to the data list
                data.append({"Person": name, "Time": time.strftime("%Y-%m-%d %H:%M:%S")})
                # create a DataFrame from the collected data
                df = pd.DataFrame(data)
                # write the DataFrame to a CSV file
                csv_file = "identified_persons.csv"
                df.to_csv(csv_file, index=False)

                person_identified = True

    # reset the timer if a person was identified
    if person_identified:
        start_time = time.time()
        person_identified = False

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # skip frames if no face is detected
    if not face_detected:
        continue

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()