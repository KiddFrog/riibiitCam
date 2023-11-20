import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to align an image to a reference image
def align_to_reference(reference_img, img_to_align):
    # Check if the images are empty
    if reference_img is None or img_to_align is None:
        print("Error: One or both images are empty.")
        return None

    # Convert images to grayscale
    gray_reference = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray_to_align = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    # Use ORB to detect and compute keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray_reference, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_to_align, None)

    # Use a matcher (e.g., BFMatcher) to find best matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Take top N matches (you may need to experiment with this value)
    num_good_matches = 50
    good_matches = matches[:num_good_matches]

    # Get corresponding points in both images
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Use findHomography to calculate the transformation matrix
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use warpPerspective to apply the transformation
    aligned_img = cv2.warpPerspective(img_to_align, M, (reference_img.shape[1], reference_img.shape[0]))

    return aligned_img

# Load your images with full paths
image2 = cv2.imread("~/Desktop/riibiit/PICTURES/image2.jpg")
image1 = cv2.imread("~/Desktop/riibiit/PICTURES/image1.jpg")
image3 = cv2.imread("~/Desktop/riibiit/PICTURES/image3.jpg")
image4 = cv2.imread("~/Desktop/riibiit/PICTURES/image4.jpg")

# Align images to image2
aligned_image1 = align_to_reference(image2, image1)
aligned_image2 = align_to_reference(image2, image2)
aligned_image3 = align_to_reference(image2, image3)
aligned_image4 = align_to_reference(image2, image4)

# Check if any of the aligned images is None
if any(img is None for img in [aligned_image1, aligned_image2, aligned_image3, aligned_image4]):
    print("Error: One or more aligned images are empty.")
    exit(1)

# Convert images to the correct data type for display
aligned_image1_display = cv2.cvtColor(aligned_image1, cv2.COLOR_BGR2RGB)
aligned_image2_display = cv2.cvtColor(aligned_image2, cv2.COLOR_BGR2RGB)
aligned_image3_display = cv2.cvtColor(aligned_image3, cv2.COLOR_BGR2RGB)
aligned_image4_display = cv2.cvtColor(aligned_image4, cv2.COLOR_BGR2RGB)

# Create a figure and axis
fig, ax = plt.subplots()

# Display the aligned images
img_display = ax.imshow(aligned_image2_display)

# Function to update the displayed image in the animation
def update(frame):
    if frame == 0:
        img_display.set_array(aligned_image1_display)
    elif frame == 1:
        img_display.set_array(aligned_image2_display)
    elif frame == 2:
        img_display.set_array(aligned_image3_display)
    elif frame == 3:
        img_display.set_array(aligned_image4_display)

# Create the animation
animation = FuncAnimation(fig, update, frames=np.arange(0, 4), interval=1000)

# Show the animation
plt.show()