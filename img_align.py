import cv2
import numpy as np

# Load the images
image1 = cv2.imread('path/to/image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('path/to/image2.jpg', cv2.IMREAD_GRAYSCALE)  # The baseline image
image3 = cv2.imread('path/to/image3.jpg', cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread('path/to/image4.jpg', cv2.IMREAD_GRAYSCALE)

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

# Display aligned images
cv2.imshow('Aligned image1', aligned_image1)
cv2.imshow('Aligned image3', aligned_image3)
cv2.imshow('Aligned image4', aligned_image4)
cv2.waitKey(0)
cv2.destroyAllWindows()
