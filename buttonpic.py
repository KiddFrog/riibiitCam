import subprocess
from gpiozero import Button
from signal import pause

# Create a Button object for capturing photos
button = Button(21)

# Set the initial value of the preview flag
preview_flag = False

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
        print("Preview stopped. Capturing photo...")

# Assign the function to the button release event
button.when_released = toggle_preview

# Keep the script running and listening for events
pause()
