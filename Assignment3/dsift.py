# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:32:40 2019

@author: 20172139

1. Perform Dense SIFT-based matching on the given pairs of images.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def generateDenseKeypoints(width,height,step_size):
#    sift = cv2.xfeatures2d.SIFT_create()

    keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(step_size,height, step_size)
                                        for x in range(step_size, width, step_size)]

    return keypoints


# Main function definition
#def main():

img = cv2.imread(r'Stereo Images\Stereo_Pair1.jpg')


height,width = img.shape[:2]
width_cutoff = width // 2
img1 = img[:, :width_cutoff]
img2 = img[:, width_cutoff:]

#img1=cv2.imread(r'Stereo Images\Stereo_Pair1_Left.jpg')
#img2=cv2.imread(r'Stereo Images\Stereo_Pair1_Right.jpg')


rgb1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
gray1 = cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY)

rgb2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
gray2 = cv2.cvtColor(rgb2,cv2.COLOR_RGB2GRAY)

h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]

step_size=40
keypoints1_dense=generateDenseKeypoints(w1,h1,step_size)
keypoints2_dense=generateDenseKeypoints(w2,h2,step_size)


x_vals1=[kp.pt[0] for kp in keypoints1_dense]
y_vals1=[kp.pt[1] for kp in keypoints1_dense]
x_vals2=[kp.pt[0] for kp in keypoints2_dense]
y_vals2=[kp.pt[1] for kp in keypoints2_dense]

plt.figure(figsize=(12,12))

plt.subplot(1,2,1)
plt.imshow(rgb1)
plt.axis('off')
plt.title('Left Image')

plt.subplot(1,2,2)
plt.imshow(rgb1)
plt.scatter(x_vals1,y_vals1, marker='+',c='b')
plt.axis('off')
plt.title('Dense Keypoints - Image1')

plt.show()

plt.figure(figsize=(12,12))

plt.subplot(1,2,1)
plt.imshow(rgb2)
plt.axis('off')
plt.title('Right Image')


plt.subplot(1,2,2)
plt.imshow(rgb2)
plt.scatter(x_vals2,y_vals2, marker='+',c='b')
plt.axis('off')
plt.title('Dense Keypoints - Image2')

plt.show()

# =============================================================================
#Dense SIFT feature means that we are calculating the SIFT descriptor for each
#and every pixel or predefined #  grid of patches in the image. So you have to
#give dense feature points (instead of what you get from the .detect() function) 
# when you call the siftExtractorObject.compute() function. 
# =============================================================================

#SIFT , Compute descriptors using dense keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints1_sift,desc1 = sift.compute(gray1,keypoints1_dense)
keypoints2_sift,desc2 = sift.compute(gray1,keypoints2_dense)


# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1,desc2, k=2)
# Apply ratio test
good_matches = []

good=[]

for m,n in matches:
    if m.distance < 0.8*n.distance:
        good_matches.append([m])
        good.append(m)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(rgb1,keypoints1_dense,rgb2,keypoints2_dense,good_matches,None,flags=2)
plt.figure(figsize=(12,12))
plt.title('Dense SIFT based matching - Stereo pair images')
plt.axis('off')
plt.imshow(img3)
plt.show()


