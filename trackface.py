import cv2
import subprocess
import numpy as np
import sys

# Open libcamera-hello subprocess
cmd = ["libcamera-hello", "-t", "5000", "-s", "1280x720", "-f", "NV12"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Load the SSD model for face detection
model_path = "opencv_face_detector_uint8.pb"
config_path = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

while True:
    # Read a frame from libcamera-hello stdout
    raw_frame = process.stdout.read(int(1280 * 720 * 1.5))
    if not raw_frame:
        break

    # Convert the NV12 frame to BGR
    frame = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw_frame, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    # Use the SSD model for face detection
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Draw bounding boxes around the detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release libcamera-hello subprocess
process.terminate()
cv2.destroyAllWindows()
