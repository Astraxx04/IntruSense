import os
import cv2
import imutils
import time
import pickle
import numpy as np
import pandas as pd
from imutils.video import FPS
from imutils.video import VideoStream


fps = None

vs = None
video_feed_active = True


# Function to generate video frames
def generate_frames():
    print("Hello")
    fps = FPS().start()
    
    # Load serialized face detector
    print("Loading Face Detector...")
    protoPath = r"face_detection_model/deploy.prototxt"
    modelPath = r"face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    vs = VideoStream(src=0).start()

    # Load serialized face embedding model
    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch(r"face_detection_model/openface_nn4.small2.v1.t7")

    # Load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(r"output/recognizer", "rb").read())
    le = pickle.loads(open(r"output/le.pickle", "rb").read())

    # Initialize the video stream, then allow the camera sensor to warm up
    print("Starting Video Stream...")
    time.sleep(2.0)

    # Start the FPS throughput estimator

    start_time = time.time()
    person_identified = False

    # Load existing data from CSV file (if it exists)
    csv_file = "identified_persons.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame()

    # DataFrame for identified persons
    data = []

    # Define a route to serve the video stream
    while video_feed_active:
        frame = vs.read()

        # Resize the frame to have a width of 600 pixels (while maintaining the aspect ratio),
        # and then grab the image dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # Construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # Initialize flag for face presence in the current frame
        face_detected = False
        person_identified = False
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            # Filter out weak detections
            if confidence >= 0.8:
                # Compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # Construct a blob for the face ROI, then pass the blob through our face embedding model
                # to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # Perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]

                # If the probability is below a certain threshold, consider it as an unknown face
                if proba < 0.8:
                    name = "Unknown"
                else:
                    name = le.classes_[j]

                # Draw the bounding box of the face along with the associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # Set the face_detected flag to True
                face_detected = True

                # Check if enough time has elapsed since the last person identification
                elapsed_time = time.time() - start_time
                print(elapsed_time)
                if elapsed_time >= 5 and not person_identified:
                    # Output the identified person
                    print("Identified Person:", name)
                    df = df.append({"Person": name, "Time": time.strftime("%Y-%m-%d %H:%M:%S")}, ignore_index=True)
                    # Write the updated DataFrame to the CSV file
                    df.to_csv(csv_file, index=False)
                    person_identified = True

        # Reset the timer if a person was identified
        if person_identified:
            start_time = time.time()
            person_identified = False

        # Update the FPS counter
        fps.update()

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cleanup()

def cleanup():
    # Cleanup
    if fps is not None:
        fps.stop()
        # print("Elapsed time: {:.2f}".format(fps.elapsed()))
        # print("Approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    if vs is not None:
        vs.stop()

