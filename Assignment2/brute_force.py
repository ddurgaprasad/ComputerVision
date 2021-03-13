import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


img1 = cv.imread(r'test_images\img2_1.png',0)          # queryImage
img2 = cv.imread(r'test_images\img2_2.png',0) # trainImage
## Initiate ORB detector
#orb = cv.ORB_create()
#print('Brute Force Matcher')
## find the keypoints and descriptors with ORB
#kp1, des1 = orb.detectAndCompute(img1,None)
#kp2, des2 = orb.detectAndCompute(img2,None)
#
## create BFMatcher object
#bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
## Match descriptors.
#matches = bf.match(des1,des2)
## Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)
## Draw first 10 matches.
#img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
#
#
#plt.title('Brute Force')
#plt.imshow(img3),plt.show()


print('SIFT detector')
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
good_without_list = []
for m,n in matches:
    if m.distance < 0.05*n.distance:
        good.append([m])
        good_without_list.append(m)
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
plt.title('SIFT ')
plt.imshow(img3),plt.show()



#
#print('FLANN detector')
## FLANN parameters
#FLANN_INDEX_KDTREE = 1
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks=50)   # or pass empty dictionary
#flann = cv.FlannBasedMatcher(index_params,search_params)
#matches = flann.knnMatch(des1,des2,k=2)
## Need to draw only good matches, so create a mask
#matchesMask = [[0,0] for i in range(len(matches))]
#
#
## ratio test as per Lowe's paper
#for i,(m,n) in enumerate(matches):
#    if m.distance < 0.0005*n.distance:
#        matchesMask[i]=[1,0]
#draw_params = dict(matchColor = (0,255,0),
#                   singlePointColor = (255,0,0),
#                   matchesMask = matchesMask,
#                   flags = 0)
#img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
#plt.title('FLANN')
#plt.imshow(img3,),plt.show()
#
#
#MIN_MATCH_COUNT=10
#
#if len(good)>MIN_MATCH_COUNT:
#    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_without_list ]).reshape(-1,1,2)
#    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_without_list ]).reshape(-1,1,2)
#
#    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
#    matchesMask = mask.ravel().tolist()
#
#    h,w = img1.shape
#    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#    dst = cv.perspectiveTransform(pts,M)
#
#    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
#
#else:
#    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
#    matchesMask = None
#
#
#draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                   singlePointColor = None,
#                   matchesMask = matchesMask, # draw only inliers
#                   flags = 2)
#
#img3 = cv.drawMatches(img1,kp1,img2,kp2,good_without_list,None,**draw_params)
#
#plt.title('What us this ?')
#plt.imshow(img3, 'gray'),plt.show()
#
#
#
#
#
#
#
#
#
#
#
