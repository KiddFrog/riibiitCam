from gpiozero import Button
from signal import pause
from time import sleep
import os
import cv2
import numpy as np
from PIL import Image
import time
import shutil

# Set the output directory for photos and aligned images
OUTPUT_DIR = os.path.expanduser("~/Desktop/riibiitCam/PICTURES")

# Set the dimensions for each camera image
WIDTH = 2328
HEIGHT = 1748

# Initialize the photo counter
photo_counter = 0

# Function to capture a photo and split it into four separate images
def capture_photo():
    global photo_counter
    photo_counter += 1

    filename = f"{time.strftime('%Y%m%d-%H%M%S')}_photo{photo_counter}"
    photo_path = os.path.join(OUTPUT_DIR, f"{filename}.jpg")

    # Capture a photo
    os.system(f"libcamera-jpeg -o {photo_path}")

    # Split the photo into four separate images
    image = Image.open(photo_path)
    for i in range(4):
        x = WIDTH * (i % 2)
        y = HEIGHT * (i // 2)
        cropped_image = image.crop((x, y, x + WIDTH, y + HEIGHT))

        # Save each cropped image with a unique name
        cropped_filename = f"{filename}_image{i + 1}.jpg"
        cropped_image.save(os.path.join(OUTPUT_DIR, cropped_filename))

    # Create a GIF from the four images
    image_paths = [os.path.join(OUTPUT_DIR, f"{filename}_image{i + 1}.jpg") for i in range(4)]
    reversed_image_paths = image_paths[::-1]

    gif_path = os.path.join(OUTPUT_DIR, f"{filename}.gif")

    with Image.open(image_paths[0]) as gif_image:
        gif_image.save(
            gif_path,
            save_all=True,
            append_images=[Image.open(path) for path in image_paths[1:]] + [Image.open(path) for path in reversed_image_paths],
            loop=0,
            duration=100
        )

    return gif_path

# Function to align images and create a looping GIF
def align_images(gif_path):
    # Load the images
    image_paths = [os.path.join(OUTPUT_DIR, f'{filename}_image{i + 1}.jpg') for i in range(4)]
    images = [cv2.imread(path) for path in image_paths]

    # Convert BGR images to RGB
    images_rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

    # Initialize ORB detector with adjusted parameters
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

    # Iterate over consecutive image pairs
    for i in range(1, len(images)):
        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(images[i - 1], None)
        kp2, des2 = orb.detectAndCompute(images[i], None)

        # Use the BFMatcher to find the best matches between descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test with adjusted threshold
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

        # Check if there are enough good matches
        if len(good_matches) < 4:
            print(f"Not enough good matches between image{i-1} and image{i} to calculate homography.")
        else:
            print(f"Generating alignment between image{i-1} and image{i}...")

            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography matrix
            homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Adjust homography matrix to preserve aspect ratio
            aspect_ratio = float(images[i].shape[1]) / float(images[i].shape[0])
            homography_matrix[0, 1] = homography_matrix[0, 1] * aspect_ratio
            homography_matrix[1, 0] = homography_matrix[1, 0] / aspect_ratio

            # Apply homography to align images without stretching
            aligned_image = cv2.warpPerspective(images_rgb[i-1], homography_matrix, (images[i].shape[1], images[i].shape[0]))
            print(f"Image{i-1} aligned with Image{i}.")

            # Save aligned image
            aligned_filename = f"{filename}_Align{i-1}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, aligned_filename), cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR))

    print("All images aligned.")

    # Create a looping GIF
    aligned_image_paths = [os.path.join(OUTPUT_DIR, f"{filename}_Align{i-1}.jpg") for i in range(1, len(images))]
    gif_path = os.path.join(OUTPUT_DIR, f"{filename}_aligned_{time.strftime('%Y%m%d-%H%M%S')}.gif")

    with Image.open(aligned_image_paths[0]) as gif_image:
        gif_image.save(
            gif_path,
            save_all=True,
            append_images=[Image.open(path).convert('RGB') for path in aligned_image_paths],
            loop=0,
            duration=100
        )

    print(f"Generated GIF: {gif_path}")

# Create a button object associated with GPIO pin 21
button = Button(21)

# Function to be executed when the button is pressed
def button_pressed():
    # Capture a photo, split it into four images, and create a GIF
    gif_path = capture_photo()

    # Align the images and create a looping GIF
    align_images(gif_path)

# Assign the button_pressed function to the when_pressed attribute of the button
button.when_pressed = button_pressed

# Keep the script running
pause()
