import cv2
import subprocess
import numpy as np
import sys

# Open libcamera-hello subprocess
cmd = ["libcamera-hello", "-t", "5000", "-s", "1280x720", "-f", "NV12"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Load the pre-trained face detector
haarcascade_path = "/home/froggo/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarcascade_path)
while True:
    # Read a frame from libcamera-hello stdout
    raw_frame = process.stdout.read(1280 * 720 * 1.5)  # Assuming resolution is 1280x720 and format is NV12
    if not raw_frame:
        break

    # Convert the NV12 frame to BGR
    frame = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw_frame, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release libcamera-hello subprocess
process.terminate()
cv2.destroyAllWindows()



Traceback (most recent call last):
  File "/home/froggo/Desktop/riibiit/trackface.py", line 15, in <module>
    raw_frame = process.stdout.read(1280 * 720 * 1.5)  # Assuming resolution is 1280x720 and format is NV12
TypeError: argument should be integer or None, not 'float'


------------------
(program exited with code: 1)
Press return to continue



