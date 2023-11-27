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

# Initialize video_process variable
video_process = None

# Function to start the video streaming
def start_video_stream():
    global video_process
    # Start the video streaming using libcamera-vid
    video_process = subprocess.Popen(["libcamera-vid", "-t", "0"])

# Function to capture a photo, create a GIF, and display the GIF
def capture_photo():
    global video_process
    if video_process:
        # If the video process is running, terminate it
        video_process.terminate()
        video_process.wait()

    # Generate a unique filename based on the current date and time
    filename = time.strftime("%Y%m%d-%H%M%S")

    # Use libcamera-vid to capture a photo and save it to the output directory
    os.system(f"libcamera-vid -o {os.path.join(OUTPUT_DIR, filename + '.jpg')} -n 1")

    # Introduce a longer delay before attempting to open the image file
    time.sleep(3)

    # Check if the image file exists
    image_path = os.path.join(OUTPUT_DIR, filename + ".jpg")
    if os.path.exists(image_path):
        # Split the photo into four separate images (one from each camera)
        image = Image.open(image_path)
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

        # Display the GIF using the default image viewer (change the command as needed)
        os.system(f"xdg-open {gif_path}")
    else:
        print(f"Image file not found: {image_path}")

    # Terminate the video streaming process
    video_process.terminate()
    video_process.wait()

# Function to be called when the button is pressed
def on_button_press():
    print("Button pressed!")
    capture_photo()

# Assign the function to the button press event
button.when_pressed = on_button_press

# Start the video streaming when the script begins
start_video_stream()

# Keep the script running and listening for events
pause()
