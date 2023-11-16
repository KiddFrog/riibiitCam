import os
import time
from PIL import Image
import numpy as np

# Set the output directory for the photos
OUTPUT_DIR = os.path.expanduser("~/Desktop/riibiit/PICTURES")

# Set the dimensions for each camera image
WIDTH = 2328
HEIGHT = 1748

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

        # Save the cropped image
        cropped_filename = f"crop_{filename}_{i}.jpg"
        cropped_image_path = os.path.join(OUTPUT_DIR, cropped_filename)
        cropped_image.save(cropped_image_path)

        tracked_images.append(cropped_image_path)

        # Save face positions
        face_positions.append((x, y))

    # Save face positions to a text file
    with open(os.path.join(OUTPUT_DIR, f"{filename}_face_positions.txt"), 'w') as f:
        for i, (x, y) in enumerate(face_positions):
            f.write(f"image_{i} x = {x}\nimage_{i} y = {y}\n")

    return tracked_images

def align_images_and_create_gif(filename, tracked_images):
    # Resize and copy images
    os.system(f"convert {tracked_images[0]} -resize 1000 +repage {os.path.join(OUTPUT_DIR, 'aligned_00.jpg')}")

    # Align images using align_image_stack
    os.system(f"align_image_stack -i -m -s 1 -C -a {os.path.join(OUTPUT_DIR, 'aligned_')} -C {os.path.join(OUTPUT_DIR, 'aligned_00.jpg')}")

    # Equalize color and brightness using PTblender
    os.system(f"PTblender -k 0 -t 0 -p {os.path.join(OUTPUT_DIR, 'aligned_')} {tracked_images[0]}")

    # Create animated GIF using convert
    gif_path = os.path.join(OUTPUT_DIR, f"aligned_gif_{filename}.gif")
    os.system(f"convert -format jpg -rotate '-90' -resize 600 +repage -delay 20 -loop 0 -colors 100 {os.path.join(OUTPUT_DIR, 'aligned_')}* {gif_path}")

    # Cleanup
    os.system(f"rm {os.path.join(OUTPUT_DIR, 'aligned_')}*")

# Step 1: Capture Photo
filename = capture_photo()

# Step 2: Crop and Track Faces
tracked_images = crop_and_track(filename)

# Step 3: Align Images and Create GIF
align_images_and_create_gif(filename, tracked_images)
