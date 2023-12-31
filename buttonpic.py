import os
import time
import subprocess
from PIL import Image
from gpiozero import Button
from signal import pause

# Set the output directory for the photos
OUTPUT_DIR = os.path.expanduser("~/Desktop/riibiit/PICTURES")

# Set the dimensions for each camera image
WIDTH = 2328
HEIGHT = 1748

# Create a Button object for capturing photos
button = Button(21)

# Set the initial value of the preview flag
preview_flag = True

def toggle_preview():
    global preview_flag
    preview_flag = not preview_flag  # Toggle the preview flag
    if preview_flag:
        # Start the video streaming using libcamera-vid with preview
        subprocess.Popen(["libcamera-vid", "-t", "0"])
        print("Preview started. Press and release the button to capture the photo and stop the preview.")
    else:
        # Stop the video streaming
        subprocess.run(["pkill", "-f", "libcamera-vid"])
        print("Preview stopped.")

def capture_photo():
    if preview_flag:
        # Generate a unique filename based on the current date and time
        filename = time.strftime("%Y%m%d-%H%M%S")

        # Use libcamera-jpeg to capture a photo and save it to the output directory
        subprocess.run(["libcamera-jpeg", "-o", f"{OUTPUT_DIR}/{filename}.jpg"])

        # Split the photo into four separate images (one from each camera)
        image = Image.open(os.path.join(OUTPUT_DIR, f"{filename}.jpg"))
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

        # Open the GIF with the default system viewer
        subprocess.run(["xdg-open", gif_path])  # Adjust this line based on your system (xdg-open is for Linux)

    else:
        print("Cannot capture photo without preview. Press and release the button to start the preview.")

# Assign the functions to the button events
button.when_released = toggle_preview
button.when_pressed = capture_photo

# Keep the script running and listening for events
pause()
