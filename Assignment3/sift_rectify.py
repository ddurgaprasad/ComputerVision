

import cv2
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1111)


#https://github.com/opencv/opencv/pull/1488/files
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1.ravel()),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2.ravel()),5,color,-1)
    return img1,img2

def getInliers(mask, num=10):
    matchesMask = mask.ravel().tolist()
    indices = []
    for ind in range(len(matchesMask)):
        if matchesMask[ind] == 1:
            indices.append(ind)
    matchesMask = [0]*len(matchesMask)
    np.random.shuffle(indices)
    indices = indices[:num]
    for ind in indices:
            matchesMask[ind] = 1
    return matchesMask

# Main function definition
#def main():

img = cv2.imread(r'Stereo Images\Stereo_Pair1.jpg')


height,width = img.shape[:2]
width_cutoff = width // 2
img1 = img[:, :width_cutoff]
img2 = img[:, width_cutoff:]


rgb1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
gray1 = cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY)

rgb2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
gray2 = cv2.cvtColor(rgb2,cv2.COLOR_RGB2GRAY)

h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]



plt.subplot(1,2,1)
plt.imshow(rgb1)
plt.axis('off')
plt.title('Left Image')


plt.subplot(1,2,2)
plt.imshow(rgb2)
plt.axis('off')
plt.title('Right Image')


plt.show()

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
interest_points1, desc1 = sift.detectAndCompute(img1,None)
interest_points2, desc2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches1 = bf.knnMatch(desc1,desc2, k=2)

matchesList = []

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
    
matchesList_Image1=[item[0:2] for item in matchesList]
pts1=np.int32(np.round(matchesList_Image1)).reshape(-1,1,2)

matchesList_Image2=[item[2:] for item in matchesList]    
pts2=np.int32(np.round(matchesList_Image2)).reshape(-1,1,2)

        
    
    
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC,1)
print(F)


matchesMask = getInliers(mask, 10)
inlierImage = cv2.drawMatches(rgb1,interest_points1,rgb2,interest_points2,
                              good_without_list,None,matchesMask = matchesMask,flags = 2)
plt.imshow(inlierImage)   
plt.show() 

# Select inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]



# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(gray1,gray2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(gray2,gray1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
    
 





















    
    