import os
import cv2
from shutil import copyfile
import numpy as np

MAX_MATCHES = 1000
GOOD_MATCH_PERCENT = 0.15

def align_images(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    if descriptors1 is None or descriptors2 is None:
        # Keypoints or descriptors not found in one of the images
        return None, None

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
    os.makedirs(output_folder, exist_ok=True)

    imReference = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    copyfile(image_paths[0], os.path.join(output_folder, "aligned_img1.JPG"))

    for i in range(2, len(image_paths)):  # Start from img2
        im = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        print(f"Aligning images img{i} and img{i + 1} ...")

        imReg, _ = align_images(im, imReference)

        if imReg is not None:
            outFilename = os.path.join(output_folder, f"aligned_img{i + 1}.JPG")
            print("Saving aligned image : ", outFilename)
            cv2.imwrite(outFilename, imReg)

            # Update reference image for the next iteration
            imReference = cv2.imread(outFilename, cv2.IMREAD_COLOR)
        else:
            print(f"Error aligning images img{i} and img{i + 1}.")

if __name__ == '__main__':
    path = "~/Desktop/riibiit/PICTURES"
    image_paths = [os.path.join(path, f"img{i}.JPG") for i in range(1, 5)]
    align_and_save_images(image_paths, os.path.join(path, "aligned"))
