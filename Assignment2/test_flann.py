import cv2
import numpy as np
from skimage import filters
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from skimage.feature import corner_peaks
from scipy import signal

plt.rcParams['figure.figsize'] = (6,8)  # set default size of plots
plt.rcParams['image.cmap'] = 'gray'

import random
random.seed(9840)

img1 = cv2.imread(r'test_images\model_chickenbroth.jpg',0)  # source
img2 = cv2.imread(r'test_images\chickenbroth_05.jpg',0) # target

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
interest_points1, desc1 = sift.detectAndCompute(img1,None)
interest_points2, desc2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches1 = bf.knnMatch(desc1,desc2, k=2)
# Sort per their distance
# Apply ratio test
#matches1 = bf.match(desc1,desc2)
#matches1 = sorted(matches1, key = lambda x:x.distance)

matchesList = []

#for match in matches1:
#    (u, v) = interest_points1[match.queryIdx].pt
#    (x, y) = interest_points2[match.trainIdx].pt
#    matchesList.append([u, v, x, y])

   

good = []
good_without_list = []
for m,n in matches1:
    if m.distance < 0.7*n.distance:
        good.append([m])
        good_without_list.append(m)
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,interest_points1,img2,interest_points2,good,None,flags=2)
plt.title('SIFT ')
plt.imshow(img3),plt.show()

for match in good_without_list:
    (u, v) = interest_points1[match.queryIdx].pt
    (x, y) = interest_points2[match.trainIdx].pt
    matchesList.append([u, v, x, y])
    

