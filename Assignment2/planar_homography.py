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


def calculateHomography(matchesList):
    #loop through correspondences and create assemble matrix
    A = []
          
    for item in matchesList:        
        #u,v are point1 (corresponding in first image)
        #x,y are point2 (corresponding in second image)
        u,v,x,y=item
        #Homogeneous point for (x,y) is (x,y,w)
#        w=1
#        w_dash=1
#        p1 =np.matrix([u,v,w_dash])
#        p2 =np.matrix([x,y,w])

        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, v*y, v])
        
    #svd composition
    U, S, V = np.linalg.svd(np.asarray(A))

    H = V[-1, :].reshape(3, 3)/V[-1, -1]

    return H


def ransacH(matchesList, num_iter=500, tol=0.6):

    num_matches=len(matchesList)
 
    # RANSAC
    bestH, most_inliers = None, 0
    
    matchesList_Image1=[item[0:2] for item in matchesList]
    xx = [item[0] for item in matchesList_Image1]
    yy=  [item[1] for item in matchesList_Image1]    
    pts1 = []
    pts1.append(xx)
    pts1.append(yy)
    pts1=np.asarray(pts1)
    
    
    matchesList_Image2=[item[2:] for item in matchesList]    
    xx = [item[0] for item in matchesList_Image2]
    yy=  [item[1] for item in matchesList_Image2]    
    pts2 = []
    pts2.append(xx)
    pts2.append(yy)
    pts2=np.asarray(pts2)
        

    
    for i in range(num_iter):
        # 4 random points are picked each time for RANSAC
        random_indices = np.random.randint(num_matches, size=4)

        random_4_matches=[]
        
        pt1=matchesList[ random_indices[0]]
        pt2=matchesList[ random_indices[1]]
        pt3=matchesList[ random_indices[2]]
        pt4=matchesList[ random_indices[3]]
        
        random_4_matches.append(pt1)
        random_4_matches.append(pt2)
        random_4_matches.append(pt3)
        random_4_matches.append(pt4)
            
        H = calculateHomography(random_4_matches)
        
        # Compute H1to2
        H_inv = np.linalg.inv(H)
        H_inv = H_inv/H_inv[-1, -1]
        
        proj = np.matmul(H_inv, np.vstack((pts1, [1]*num_matches)))
        proj = proj/proj[-1, :]
        
        # Calculate back projection error and number of inliers within tolerance
        distances = np.linalg.norm(proj.T - np.vstack((pts2, [1]*num_matches)).T, axis=1)
        inliers = len(distances[distances<tol])
        
        if inliers > most_inliers:
            bestH, most_inliers = H, inliers
  
    return bestH



img1 = cv2.imread(r'test_images\model_chickenbroth.jpg',0)  # source
img2 = cv2.imread(r'test_images\chickenbroth_05.jpg',0) # target

# SIFT detector
num_of_corners=30

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
interest_points1, desc1 = sift.detectAndCompute(img1,None)
interest_points2, desc2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches1 = bf.knnMatch(desc1,desc2, k=2)
print("Sift matches1 count" ,len(matches1) )
## Apply ratio test
#matches1 = bf.match(desc1,desc2)
## Sort per their distance
#matches1 = sorted(matches1, key = lambda x:x.distance)
#
matchesList = []
#
#for match in matches1:
#    (u, v) = interest_points1[match.queryIdx].pt
#    (x, y) = interest_points2[match.trainIdx].pt
#    matchesList.append([u, v, x, y])
#
## cv.drawMatchesKnn expects list of lists as matches.
#img3 = cv2.drawMatches(img1,interest_points1,img2,interest_points2,matches1,None,flags=2)
#plt.axis('off')
#plt.title('SIFT with best match using KNN')
#plt.imshow(img3)
#plt.show()
#   



good = []
good_without_list = []
for m,n in matches1:
    if m.distance < 0.7*n.distance:
        good.append([m])
        good_without_list.append(m)
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,interest_points1,img2,interest_points2,good,None,flags=2)
plt.title('SIFT ,Good Matched ')
plt.imshow(img3),plt.show()

for match in good_without_list:
    (u, v) = interest_points1[match.queryIdx].pt
    (x, y) = interest_points2[match.trainIdx].pt
    matchesList.append([u, v, x, y])
    
    
#H=calculateHomography(matchesList)
H=ransacH(matchesList, 1000, 2)
print(H)


plt.subplot(1,3,1)
plt.axis('off')
plt.title('Image1')
plt.imshow(img1)

plt.subplot(1,3,2)
plt.axis('off')
plt.title('Image2')
plt.imshow(img2)

plt.subplot(1,3,3)
img_dst=cv2.warpPerspective(img2,H,(img1.shape[1],img1.shape[0]))
plt.axis('off')
plt.title('Warped Image')
plt.imshow(img_dst)
plt.show()





