# CSCI507 - Computer Vision
# Final Project
# Eric Klatzco & Samuel Pearson
# mask.py - responsible for detecting just the mask
# This mask file will try two different ways
# The first is simple binary imaging / connected components
# Afterwards, we'll be trying to do object detection specifically with the wrinkles of a mask

#

import cv2
import glob
import numpy as np
import os

MASK_IMAGE_DIRECTORY = "masks"


# Binary Imaging
# Work referenced from Homework Two/Three, Lab 4, along with the image looping
def binaryDetection():
    # Image looping setup
    assert (os.path.exists(MASK_IMAGE_DIRECTORY))
    image_file_names = glob.glob(os.path.join(MASK_IMAGE_DIRECTORY, "*.jpg"))
    assert (len(image_file_names) > 0)

    # Image looping
    for image_file_name in image_file_names:
        bgr_image = cv2.imread(image_file_name)
        cv2.imshow("image", bgr_image)

        # Gray image + thresholding with Otsu's & our inverse
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        thresh, binary_img = cv2.threshold(gray_image,
                                           thresh=0,
                                           maxval=255,
                                           type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv_binary_img = cv2.bitwise_not(binary_img)
        # Trying both MORPH_ELLIPSE and MORPH_RECT
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(binary_img)
        cv2.imshow("bin", binary_img)

        inv_binary_img = cv2.morphologyEx(inv_binary_img, cv2.MORPH_OPEN, kernel)
        inv_binary_img = cv2.morphologyEx(inv_binary_img, cv2.MORPH_CLOSE, kernel)
        inv_num_labels, inv_labels_img, inv_stats, inv_centroids = cv2.connectedComponentsWithStats(inv_binary_img)
        cv2.imshow("Inv", inv_binary_img)

        bgr_image_display = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

        for stat in stats:
            for inv_stat in inv_stats:
                x0 = stat[cv2.CC_STAT_LEFT]
                x1 = inv_stat[cv2.CC_STAT_LEFT]
                y0 = stat[cv2.CC_STAT_TOP]
                y1 = inv_stat[cv2.CC_STAT_TOP]
                # if ((1 < y0 - y1 < 6) and
                #         (1 < x0 - x1 < 6)):
                bgr_image_display = cv2.rectangle(img=bgr_image, pt1=(x0 + 2, y0 + 2), pt2=(x0 + 3, y0 + 3),
                                                  color=(0, 0, 255),
                                                  thickness=2)
        cv2.imshow("out", bgr_image_display)

        # Quick exiting
        key_pressed = cv2.waitKey(0) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC
