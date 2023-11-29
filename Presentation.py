import os
import time
from PIL import Image
import cv2
import numpy as np
from gpiozero import Button
from signal import pause

# Set the output directory for the photos
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

    # Open the first GIF
    os.system(f"open {gif_path}")

    # Call the second code snippet
    align_images()

def align_images():
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

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)  # The baseline image
    kp3, des3 = orb.detectAndCompute(image3, None)
    kp4, des4 = orb.detectAndCompute(image4, None)

    # Use the BFMatcher to find the best matches between descriptors
    bf = cv2.BFMatcher()
    matches1to2 = bf.knnMatch(des1, des2, k=2)
    matches3to2 = bf.knnMatch(des3, des2, k=2)
    matches4to2 = bf.knnMatch(des4, des2, k=2)

    # Apply ratio test with adjusted threshold
    good_matches1to2 = []
    for m, n in matches1to2:
        if m.distance < 0.6 * n.distance:
            good_matches1to2.append(m)

    good_matches3to2 = []
    for m, n in matches3to2:
        if m.distance < 0.6 * n.distance:
            good_matches3to2.append(m)

    good_matches4to2 = []
    for m, n in matches4to2:
        if m.distance < 0.6 * n.distance:
            good_matches4to2.append(m)

    # Check if there are enough good matches
    if len(good_matches1to2) < 4 or len(good_matches3to2) < 4 or len(good_matches4to2) < 4:
        print("Not enough good matches to calculate homography.")
    else:
        print("Generating alignment...")

        # Extract location of good matches
        src_pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches1to2]).reshape(-1, 1, 2)
        dst_pts1 = np.float32([kp2[m.trainIdx].pt for m in good_matches1to2]).reshape(-1, 1, 2)

        src_pts3 = np.float32([kp3[m.queryIdx].pt for m in good_matches3to2]).reshape(-1, 1, 2)
        dst_pts3 = np.float32([kp2[m.trainIdx].pt for m in good_matches3to2]).reshape(-1, 1, 2)

        src_pts4 = np.float32([kp4[m.queryIdx].pt for m in good_matches4to2]).reshape(-1, 1, 2)
        dst_pts4 = np.float32([kp2[m.trainIdx].pt for m in good_matches4to2]).reshape(-1, 1, 2)

        # Compute homography matrices
        homography_matrix1, _ = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)
        homography_matrix3, _ = cv2.findHomography(src_pts3, dst_pts3, cv2.RANSAC, 5.0)
        homography_matrix4, _ = cv2.findHomography(src_pts4, dst_pts4, cv2.RANSAC, 5.0)

        # Adjust homography matrices to preserve aspect ratio
        aspect_ratio = float(image2.shape[1]) / float(image2.shape[0])
        homography_matrix1[0, 1] = homography_matrix1[0, 1] * aspect_ratio
        homography_matrix1[1, 0] = homography_matrix1[1, 0] / aspect_ratio

        homography_matrix3[0, 1] = homography_matrix3[0, 1] * aspect_ratio
        homography_matrix3[1, 0] = homography_matrix3[1, 0] / aspect_ratio

        homography_matrix4[0, 1] = homography_matrix4[0, 1] * aspect_ratio
        homography_matrix4[1, 0] = homography_matrix4[1, 0] / aspect_ratio

        # Apply homography to align images without stretching
        aligned_image1 = cv2.warpPerspective(image1_rgb, homography_matrix1, (image2.shape[1], image2.shape[0]))
        print("Image1 aligned.")

        aligned_image3 = cv2.warpPerspective(image3_rgb, homography_matrix3, (image2.shape[1], image2.shape[0]))
        print("Image3 aligned.")

        aligned_image4 = cv2.warpPerspective(image4_rgb, homography_matrix4, (image2.shape[1], image2.shape[0]))
        print("Image4 aligned.")

        # Save aligned images
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'Align1.jpg'), cv2.cvtColor(aligned_image1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'Align3.jpg'), cv2.cvtColor(aligned_image3, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'Align4.jpg'), cv2.cvtColor(aligned_image4, cv2.COLOR_RGB2BGR))

        print("All images aligned.")

        # Create a looping GIF
        aligned_image_paths = [os.path.join(OUTPUT_DIR, f'Align{i}.jpg') for i in [1, 2, 3, 4, 3, 2]]
        gif_path = os.path.join(OUTPUT_DIR, f'aligned_{time.strftime("%Y%m%d-%H%M%S")}.gif')

        with Image.open(aligned_image_paths[0]) as gif_image:
            gif_image.save(gif_path, save_all=True, append_images=[Image.open(path).convert('RGB') for path in aligned_image_paths[1:]], loop=0, duration=100)

        # Open the second GIF
        os.system(f"open {gif_path}")

# Button setup
button = Button(21)

def press():
    print("Button pressed!")
    capture_photo()  # Call the function when the button is pressed

button.when_pressed = press

pause()
