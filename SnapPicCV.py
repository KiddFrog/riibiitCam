import os
import time
from PIL import Image
import cv2

# Set the output directory for the photos
OUTPUT_DIR = os.path.expanduser("~/Desktop/PICTURES")

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

    # Run the align_gif function
    align_gif(gif_path)

def align_gif(gif_path):
    # Read the GIF using PIL
    gif = Image.open(gif_path)

    # Convert the first frame to grayscale for face detection
    first_frame = gif.convert("L")

    # Use OpenCV to detect the face in the first frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(cv2.cvtColor(np.array(first_frame), cv2.COLOR_GRAY2BGR), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the face coordinates
        x, y, w, h = faces[0]

        # Loop through all frames after the first frame and align them
        aligned_frames = []
        for frame in gif:
            # Convert the frame to grayscale
            gray_frame = frame.convert("L")

            # Crop and align the frame based on the face coordinates
            aligned_frame = gray_frame.crop((x, y, x + w, y + h)).resize((w, h))

            # Add the aligned frame to the list
            aligned_frames.append(aligned_frame)

        # Create a new GIF from the aligned frames
        aligned_gif_path = os.path.join(OUTPUT_DIR, f"wiggle_{time.strftime('%Y%m%d-%H%M%S')}.gif")
        aligned_frames[0].save(aligned_gif_path, save_all=True, append_images=aligned_frames[1:], loop=0, duration=100)

        print(f"Aligned GIF created: {aligned_gif_path}")
    else:
        print("No face detected in the first frame of the GIF.")

# Capture a single photo and then align the resulting GIF
capture_photo()
