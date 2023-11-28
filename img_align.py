import cv2
import numpy as np
import os

# Set the output directory for aligned images and the GIF
OUTPUT_DIR = os.path.expanduser("~/Desktop/riibiitCam/PICTURES")

# Load the images
image1 = cv2.imread(os.path.join(OUTPUT_DIR, 'image1.jpg'), cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(os.path.join(OUTPUT_DIR, 'image2.jpg'), cv2.IMREAD_GRAYSCALE)  # The baseline image
image3 = cv2.imread(os.path.join(OUTPUT_DIR, 'image3.jpg'), cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread(os.path.join(OUTPUT_DIR, 'image4.jpg'), cv2.IMREAD_GRAYSCALE)

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)
kp3, des3 = orb.detectAndCompute(image3, None)
kp4, des4 = orb.detectAndCompute(image4, None)

# Use the BFMatcher to find the best matches between descriptors
bf = cv2.BFMatcher()
matches1to2 = bf.knnMatch(des1, des2, k=2)
matches3to2 = bf.knnMatch(des3, des2, k=2)
matches4to2 = bf.knnMatch(des4, des2, k=2)

# Apply ratio test
good_matches1to2 = []
for m, n in matches1to2:
    if m.distance < 0.75 * n.distance:
        good_matches1to2.append(m)

good_matches3to2 = []
for m, n in matches3to2:
    if m.distance < 0.75 * n.distance:
        good_matches3to2.append(m)

good_matches4to2 = []
for m, n in matches4to2:
    if m.distance < 0.75 * n.distance:
        good_matches4to2.append(m)

# Check if there are enough good matches
if len(good_matches1to2) < 4 or len(good_matches3to2) < 4 or len(good_matches4to2) < 4:
    print("Not enough good matches to calculate homography.")
else:
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

    # Apply homography to align images
    aligned_image1 = cv2.warpPerspective(image1, homography_matrix1, (image2.shape[1], image2.shape[0]))
    aligned_image3 = cv2.warpPerspective(image3, homography_matrix3, (image2.shape[1], image2.shape[0]))
    aligned_image4 = cv2.warpPerspective(image4, homography_matrix4, (image2.shape[1], image2.shape[0]))

    # Save aligned images
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'Align1.jpg'), aligned_image1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'Align3.jpg'), aligned_image3)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'Align4.jpg'), aligned_image4)

    # Create a looping GIF
    aligned_image_paths = [os.path.join(OUTPUT_DIR, f'Align{i}.jpg') for i in [1, 3, 4, 3]]
    gif_path = os.path.join(OUTPUT_DIR, 'aligned_images.gif')

    images = [cv2.imread(path) for path in aligned_image_paths]
    gif_images = images + images[::-1]  # Reverse the order for the looping effect

    cv2.imwrite(gif_path, gif_images[0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    with cv2.VideoWriter(gif_path, cv2.VideoWriter_fourcc(*"MJPG"), 1, (images[0].shape[1], images[0].shape[0])) as video:
        for img in gif_images:
            video.write(img)

    # Display aligned images
    cv2.imshow('Aligned image1', aligned_image1)
    cv2.imshow('Aligned image3', aligned_image3)
    cv2.imshow('Aligned image4', aligned_image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
