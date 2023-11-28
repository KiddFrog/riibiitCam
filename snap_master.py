import os
import time
from PIL import Image
import cv2
import numpy as np

# Set the output directory for the photos and aligned images
OUTPUT_DIR = os.path.expanduser("~/Desktop/riibiitCam/PICTURES")

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

        # Save each cropped image as image1.jpg, image2.jpg, etc.
        cropped_filename = f"image{i + 1}.jpg"
        cropped_image.save(os.path.join(OUTPUT_DIR, cropped_filename))

    # Create a GIF from the four images
    image_paths = [os.path.join(OUTPUT_DIR, f"image{i + 1}.jpg") for i in range(4)]
    reversed_image_paths = image_paths[::-1]  # Reverse the order of images

    gif_path = os.path.join(OUTPUT_DIR, f"{filename}.gif")

    with Image.open(image_paths[0]) as gif_image:
        gif_image.save(gif_path, save_all=True, append_images=[Image.open(path) for path in image_paths[1:]] + [Image.open(path) for path in reversed_image_paths], loop=0, duration=100)

    return gif_path

def align_images(gif_path):
    # Load the images
    image1 = cv2.imread(os.path.join(OUTPUT_DIR, 'image1.jpg'))
    image2 = cv2.imread(os.path.join(OUTPUT_DIR, 'image2.jpg'))  # The baseline image
    image3 = cv2.imread(os.path.join(OUTPUT_DIR, 'image3.jpg'))
    image4 = cv2.imread(os.path.join(OUTPUT_DIR, 'image4.jpg'))

    # Duplicate image2 and save as Align2
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'Align2.jpg'), image2)

    # Convert BGR images to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3_rgb = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    image4_rgb = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)

    # Initialize ORB detector with adjusted parameters
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

    # Rest of the alignment code...

    # (Assuming you have the alignment code from image_align2.py here)

    print("All images aligned.")

    # Create a looping GIF
    aligned_image_paths = [os.path.join(OUTPUT_DIR, f'Align{i}.jpg') for i in [1, 2, 3, 4, 3, 2]]
    aligned_gif_path = os.path.join(OUTPUT_DIR, f'aligned_{time.strftime("%Y%m%d-%H%M%S")}.gif')

    with Image.open(aligned_image_paths[0]) as gif_image:
        gif_image.save(aligned_gif_path, save_all=True, append_images=[Image.open(path).convert('RGB') for path in aligned_image_paths[1:]], loop=0, duration=100)

    print(f"Generated aligned GIF: {aligned_gif_path}")

    # Open the generated GIF
    os.system(f"open {aligned_gif_path}")

if __name__ == "__main__":
    gif_path = capture_photo()
    align_images(gif_path)
