import cv2
import picamera
import picamera.array

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the PiCamera
with picamera.PiCamera() as camera:
    # Set the resolution (adjust as needed)
    camera.resolution = (640, 480)

    # Use the picamera.array module for efficient capture to OpenCV format
    with picamera.array.PiRGBArray(camera) as output:
        while True:
            # Capture a frame from the camera
            camera.capture(output, 'bgr')
            img = output.array

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display the image
            cv2.imshow('img', img)

            # Clear the stream for the next frame
            output.truncate(0)

            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

# Destroy all OpenCV windows
cv2.destroyAllWindows()
