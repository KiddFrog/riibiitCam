import os
import cv2
import numpy as np
from shutil import copyfile

MAX_MATCHES = 1000
GOOD_MATCH_PERCENT = 0.15

def align_images(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Convert descriptors to np.float32
    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h, rigid_mask = cv2.estimateAffinePartial2D(points1, points2)
    height, width, channels = im2.shape
    im1Reg = cv2.warpAffine(im1, h, (width, height))

    return im1Reg, h

def align_and_save_images(image_paths, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    imReference = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    copyfile(image_paths[0], os.path.join(output_folder, "aligned_img1.JPG"))

    for i in range(1, len(image_paths)):
        im = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        print(f"Aligning images img{i} and img{i + 1} ...")

        imReg, _ = align_images(im, imReference)

        outFilename = os.path.join(output_folder, f"aligned_img{i + 1}.JPG")
        print("Saving aligned image : ", outFilename)
        cv2.imwrite(outFilename, imReg)

        # Update reference image for the next iteration
        imReference = cv2.imread(outFilename, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    pictures_folder = os.path.expanduser("~/Desktop/riibiit/PICTURES")
    image_paths = [os.path.join(pictures_folder, f"image{i}.jpg") for i in range(1, 5)]
    align_and_save_images(image_paths, os.path.join(pictures_folder, "aligned"))
