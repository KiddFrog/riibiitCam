import os
import time
import numpy as np
from PIL import Image
import cv2

# Set the output directory for the photos
OUTPUT_DIR = os.path.expanduser("~/Desktop/riibiit/PICTURES")

# Set the dimensions for each camera image
WIDTH = 2328
HEIGHT = 1748

# Load the cascade for facial tracking
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def capture_photo():
    # Generate a unique filename based on the current date and time
    filename = time.strftime("%Y%m%d-%H%M%S")

    # Use libcamera-jpeg to capture a photo and save it to the output directory
    os.system(f"libcamera-jpeg -o {os.path.join(OUTPUT_DIR, filename + '.jpg')}")

    return filename

def crop_and_track(filename):
    # Load the captured photo
    image_path = os.path.join(OUTPUT_DIR, filename + ".jpg")
    image = Image.open(image_path)

    # Initialize a list to store tracked images
    tracked_images = []

    # Variables to store the offset of the first detected face in the second image
    offset_x = None
    offset_y = None

    for i in range(4):
        # Crop the image based on the camera's position
        x = WIDTH * (i % 2)
        y = HEIGHT * (i // 2)
        cropped_image = image.crop((x, y, x + WIDTH, y + HEIGHT))

        # Convert to OpenCV format for facial tracking
        cv_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

        # Detect faces in the cropped image
        faces = face_cascade.detectMultiScale(cv_image, 1.1, 4)

        # Adjust the positions of bounding boxes based on the offset
        if i == 1 and len(faces) > 0:
            offset_x, offset_y, _, _ = faces[0]

        for (x, y, w, h) in faces:
            x -= offset_x if offset_x is not None else 0
            y -= offset_y if offset_y is not None else 0

            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert back to PIL format
        tracked_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Save the tracked image with a new filename
        tracked_filename = f"Tracked_{filename}_{i}.jpg"
        tracked_image_path = os.path.join(OUTPUT_DIR, tracked_filename)
        tracked_image.save(tracked_image_path)

        tracked_images.append(tracked_image_path)

    return tracked_images

def create_gif(tracked_images):
    # Create a GIF from the tracked images
    gif_path = os.path.join(OUTPUT_DIR, "tracked_faces.gif")

    with Image.open(tracked_images[0]) as gif_image:
        gif_image.save(gif_path, save_all=True, append_images=[Image.open(path) for path in tracked_images[1:]], loop=0, duration=100)

# Step 1: Capture Photo
filename = capture_photo()

# Step 2: Crop and Track Faces
tracked_images = crop_and_track(filename)

# Step 3: Create GIF from Tracked Images
create_gif(tracked_images)
