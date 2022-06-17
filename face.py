# CSCI507 - Computer Vision
# Final Project
# Eric Klatzco & Samuel Pearson
# face.py - responsible for detecting if face is wearing a mask


# do some object detection for faces

import cv2
import glob
import numpy as np
import os

FACE_IMAGE_DIRECTORY = "faces"
TRAINING_IMAGE_NAME = "faces/img00.jpg"


# Object Detection Mainly
# Work referenced from Homework Five, Lab 9
# This function will try and use ORB to detect a mask on someone's face
def maskFaceDetection():
    # Image looping setup
    assert (os.path.exists(FACE_IMAGE_DIRECTORY))
    image_file_names = glob.glob(os.path.join(FACE_IMAGE_DIRECTORY, "*.jpg"))
    assert (len(image_file_names) > 0)

    # Start up orb
    orb = cv2.ORB_create(nfeatures=2000)

    bgr_train = cv2.imread(TRAINING_IMAGE_NAME)
    cv2.imshow("Training image", bgr_train)
    bgr_train_bw = cv2.cvtColor(bgr_train, cv2.COLOR_BGR2GRAY)
    kp_train, desc_train = orb.detectAndCompute(bgr_train_bw, None)

    # Image looping
    for image_file_name in image_file_names:
        bgr_query = cv2.imread(image_file_name)
        cv2.imshow("query", bgr_query)

        bgr_query_bw = cv2.cvtColor(bgr_query, cv2.COLOR_BGR2GRAY)
        kp_query, desc_query = orb.detectAndCompute(bgr_query_bw, None)

        # Matching features
        # Need to do NORM_HAMMING for orb
        matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(desc_query, desc_train, k=2)

        good = []
        for m, n in matches:
            if m.distance < .75 * n.distance:
                good.append(m)
        matches = good
        # print("Number of raw matches between training and query: ", len(matches))
        bgr_matches = cv2.drawMatches(
            img1=bgr_query, keypoints1=kp_query,
            img2=bgr_train, keypoints2=kp_train,
            matches1to2=matches, matchesMask=None, outImg=None)
        cv2.imshow("All matches", bgr_matches)

        # RANSAC / Affine
        # Calculate an affine transformation from the training image to the query image.
        A_train_query, inliers = calc_affine_transformation(matches, kp_train, kp_query)

        # Apply the affine warp to warp the training image to the query image.
        if A_train_query is not None and sum(inliers) > 7:
            # Object detected! Warp the training image to the query image and blend the images.
            # print("Object detected! Found %d inlier matches" % sum(inliers))
            warped_training = cv2.warpAffine(
                src=bgr_train, M=A_train_query,
                dsize=(bgr_query.shape[1], bgr_query.shape[0]))

            # Blend the images.
            blended_image = bgr_query / 2
            blended_image[:, :, 1] += warped_training[:, :, 1] / 2
            blended_image[:, :, 2] += warped_training[:, :, 2] / 2
            cv2.imshow("Blended", blended_image.astype(np.uint8))

            # # Affine our points  only if an object is detected
            # nP1 = A_train_query @ np.array([p1[0], p1[1], 1])
            # nP2 = A_train_query @ np.array([p2[0], p2[1], 1])
            # nP3 = A_train_query @ np.array([p3[0], p3[1], 1])
            #
            # cv2.drawMarker(bgr_query, (int(nP1[0]), int(nP1[1])), color=(255, 0, 0), markerType=cv2.MARKER_DIAMOND,
            #                thickness=2)
            # cv2.drawMarker(bgr_query, (int(nP2[0]), int(nP2[1])), color=(255, 0, 0), markerType=cv2.MARKER_DIAMOND,
            #                thickness=2)
            # cv2.drawMarker(bgr_query, (int(nP3[0]), int(nP3[1])), color=(255, 0, 0), markerType=cv2.MARKER_DIAMOND,
            #                thickness=2)
            # print(image_file_name)
            cv2.imshow("new query image", bgr_query)

        # Quick exiting
        key_pressed = cv2.waitKey(0) & 0xFF
        if key_pressed == 27:
            break  # Quit on ESC

def calc_affine_transformation(matches_in_cluster, kp_train, kp_query):
    if len(matches_in_cluster) < 3:
        # Not enough matches to calculate affine transformation.
        return None, None

    src_pts = np.float32([kp_train[m.trainIdx].pt for m in matches_in_cluster]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_query[m.queryIdx].pt for m in matches_in_cluster]).reshape(-1, 1, 2)

    A_train_query, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3,  # Default = 3
        maxIters=2000,  # Default = 2000
        confidence=0.99,  # Default = 0.99
        refineIters=10
    )
    return A_train_query, inliers
