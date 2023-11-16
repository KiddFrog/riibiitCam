import os
import time
from PIL import Image

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

def crop_images(filename):
    # Load the captured photo
    image_path = os.path.join(OUTPUT_DIR, filename + ".jpg")
    image = Image.open(image_path)

    # Initialize a list to store cropped images
    cropped_images = []

    for i in range(4):
        # Crop the image based on the camera's position
        x = WIDTH * (i % 2)
        y = HEIGHT * (i // 2)
        cropped_image = image.crop((x, y, x + WIDTH, y + HEIGHT))

        # Save the cropped image
        cropped_filename = f"crop_{filename}_{i}.jpg"
        cropped_image_path = os.path.join(OUTPUT_DIR, cropped_filename)
        cropped_image.save(cropped_image_path)

        cropped_images.append(cropped_image_path)

    return cropped_images

def create_gif(images, gif_path):
    # Reverse the order of images
    reversed_images = list(reversed(images))

    # Create animated GIF using convert
    os.system(f"convert -format jpg -rotate '-90' -resize 600 +repage -delay 20 -loop 0 -colors 100 {' '.join(reversed_images)} {gif_path}")

# Step 1: Capture Photo
filename = capture_photo()

# Step 2: Crop Images
cropped_images = crop_images(filename)

# Step 3: Create GIF with Backward Looping
gif_path = os.path.join(OUTPUT_DIR, f"animated_gif_{filename}.gif")
create_gif(cropped_images, gif_path)
