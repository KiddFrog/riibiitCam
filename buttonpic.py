import subprocess
from gpiozero import Button
from PIL import Image
from signal import pause

# Create a Button object for capturing photos
button = Button(21)

# Set the output directory for the photos
OUTPUT_DIR = "/home/pi/Desktop/riibiit/PICTURES"

# Set the dimensions for each camera image
WIDTH = 2328
HEIGHT = 1748

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

        print(f"Photo captured and saved as {filename}.jpg")
    else:
        print("Cannot capture photo without preview. Press and release the button to start the preview.")

# Assign the functions to the button events
button.when_released = toggle_preview
button.when_pressed = capture_photo

# Keep the script running and listening for events
pause()
