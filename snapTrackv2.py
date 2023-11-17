import os
import time
from PIL import Image
import cv2
import numpy as np

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

    # Initialize a list to store tracked images and face positions
    tracked_images = []
    face_positions = []

    for i in range(4):
        # Crop the image based on the camera's position
        x = WIDTH * (i % 2)
        y = HEIGHT * (i // 2)
        cropped_image = image.crop((x, y, x + WIDTH, y + HEIGHT))

        # Convert to OpenCV format for facial tracking
        cv_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

        # Detect faces in the cropped image
        faces = face_cascade.detectMultiScale(cv_image, 1.1, 4)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Save face positions
            face_positions.append((x, y, w, h))

        # Convert back to PIL format
        tracked_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Save the tracked image with a new filename
        tracked_filename = f"Tracked_img{i + 1}_{filename}.jpg"
        tracked_image_path = os.path.join(OUTPUT_DIR, tracked_filename)
        tracked_image.save(tracked_image_path)
        
        tracked_images.append(tracked_image_path)

    # Save face positions to a text file
    with open(os.path.join(OUTPUT_DIR, f"face_positions_img_{filename}.txt"), 'w') as f:
        for i, (x, y, w, h) in enumerate(face_positions):
            f.write(f"image_{i} x = {x}\nimage_{i} y = {y}\n")

    return tracked_images

def place_faces(filename, tracked_images):
    # Load face positions from the text file
    face_positions = []
    with open(os.path.join(OUTPUT_DIR, f"face_positions_img_{filename}.txt"), 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            x = int(lines[i + 1].split()[-1])
            y = int(lines[i + 2].split()[-1])
            face_positions.append((x, y))

    # Load the second tracked image for reference
    reference_image = Image.open(tracked_images[1])

    # Initialize a list to store placed images
    placed_images = []

    for i, (x, y) in enumerate(face_positions):
        # Load the corresponding tracked image
        tracked_image = Image.open(tracked_images[i])

        # Calculate the offset to align faces with the second tracked image
        offset_x = x - face_positions[1][0]
        offset_y = y - face_positions[1][1]

        # Create a new image with the aligned face
        placed_image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
        placed_image.paste(tracked_image, (offset_x, offset_y))

        # Save the placed image with a new filename
        placed_filename = f"Placed_img{i + 1}_{filename}.jpg"
        placed_image_path = os.path.join(OUTPUT_DIR, placed_filename)
        placed_image.save(placed_image_path)

        placed_images.append(placed_image_path)

    return placed_images

def create_gif(images, gif_path):
    # Create a GIF from the images
    with Image.open(images[0]) as gif_image:
        gif_image.save(gif_path, save_all=True, append_images=[Image.open(path) for path in images[1:]], loop=0, duration=100)

# Step 1: Capture Photo
filename = capture_photo()

# Step 2: Crop and Track Faces
tracked_images = crop_and_track(filename)

# Step 3: Place Faces
placed_images = place_faces(filename, tracked_images)

# Step 4: Create GIF from Placed Images
create_gif(placed_images, os.path.join(OUTPUT_DIR, f"placed_faces_{filename}.gif"))
