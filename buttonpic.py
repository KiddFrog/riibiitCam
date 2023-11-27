import os
import time
import subprocess
from PIL import Image
from gpiozero import Button

# Set the output directory for the photos
OUTPUT_DIR = os.path.expanduser("~/Desktop/riibiit/PICTURES")

# Set the dimensions for each camera image
WIDTH = 2328
HEIGHT = 1748

# Set the initial value of the preview flag
preview_flag = 0

# Create a Button object for capturing photos
button = Button(21)

def toggle_preview():
    global preview_flag
    preview_flag = 1

# Assign the function to the button press event
button.when_pressed = toggle_preview

def capture_photo():
    global preview_flag

    # Start the video streaming using libcamera-vid with preview if preview_flag is 0
    if preview_flag == 0:
        video_process = subprocess.Popen(["libcamera-vid", "-t", "0"])
        print("Preview started. Press the button to capture the photo and stop the preview.")
    else:
        # Stop the video streaming
        video_process.terminate()
        video_process.wait()
        print("Preview stopped. Capturing photo...")

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

            # Save each cropped image as image1.jpg, image2.jpg, etc.
            cropped_filename = f"image{i + 1}.jpg"
            cropped_image.save(os.path.join(OUTPUT_DIR, cropped_filename))

        # Create a GIF from the four images
        image_paths = [os.path.join(OUTPUT_DIR, f"image{i + 1}.jpg") for i in range(4)]
        reversed_image_paths = image_paths[::-1]  # Reverse the order of images

        gif_path = os.path.join(OUTPUT_DIR, f"{filename}.gif")

        with Image.open(image_paths[0]) as gif_image:
            gif_image.save(gif_path, save_all=True, append_images=[Image.open(path) for path in image_paths[1:]] + [Image.open(path) for path in reversed_image_paths], loop=0, duration=100)

        print(f"Photo captured and saved as {filename}.gif")

# Capture a single photo and then exit
capture_photo()
