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

    # Split the photo into four separate images (one from each camera)
    image = Image.open(os.path.join(OUTPUT_DIR, filename + ".jpg"))
    for i in range(4):
        x = WIDTH * (i % 2)
        y = HEIGHT * (i // 2)
        cropped_image = image.crop((x, y, x + WIDTH, y + HEIGHT))
        cropped_image.save(os.path.join(OUTPUT_DIR, f"{filename}_{i}.jpg"))

    # Create a GIF from the four images
    image_paths = [os.path.join(OUTPUT_DIR, f"{filename}_{i}.jpg") for i in range(4)]
    gif_path = os.path.join(OUTPUT_DIR, f"{filename}.gif")

    with Image.open(image_paths[0]) as gif_image:
        gif_image.save(gif_path, save_all=True, append_images=[Image.open(path) for path in image_paths[1:]], loop=0, duration=100)

# Capture a single photo and then exit
capture_photo()
