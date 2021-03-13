import cv2
import numpy as np
from matplotlib import pyplot as plt




# Read image
#im = cv2.imread(r"C:\SAI\IIIT\2019_Spring\Assignment1\CamCal_Tutorial-master\data\test.jpg", cv2.IMREAD_GRAYSCALE)
 
 
im = cv2.imread(r"C:\SAI\IIIT\2019_Spring\Assignment1\Assignment1_Data\up.jpg", cv2.IMREAD_GRAYSCALE)

height, width = im.shape[:2]

## Let's get the starting pixel coordiantes (top left of cropped top)
#start_row, start_col = int(0), int(0)
## Let's get the ending pixel coordinates (bottom right of cropped top)
#end_row, end_col = int(height * .5), int(width)
#cropped_top = im[start_row:end_row , start_col:end_col]
#print (start_row, end_row )
#print (start_col, end_col)
#
#plt.axis("off")
#plt.imshow(cropped_top)
#plt.show()
#
## Let's get the starting pixel coordiantes (top left of cropped bottom)
#start_row, start_col = int(height * .5), int(0)
## Let's get the ending pixel coordinates (bottom right of cropped bottom)
#end_row, end_col = int(height), int(width)
#cropped_bot = im[start_row:end_row , start_col:end_col]
#print (start_row, end_row )
#print (start_col, end_col)
#
#plt.axis("off")
#plt.imshow(cropped_bot)
#plt.show()

  
#scaling_factor=0.6
#
#im = cv2.resize(im1, (0,0), fx=scaling_factor, fy=scaling_factor) 

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.


params.filterByCircularity = False;
params.filterByConvexity = False;
params.filterByInertia = False;


# Set up the detector with default parameters.
is_v2 = cv2.__version__.startswith("2.")
if is_v2:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)


    
# Detect blobs.
keypoints1 = detector.detect(im)
 


#params.minArea = 1500
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)

im_with_keypoints = cv2.drawKeypoints(im, keypoints1, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Show keypoints
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)
#plt.imshow(im_with_keypoints, cmap="gray")
#plt.show()
print("keypoints1 - Points" ,len(keypoints1) )
img = im.copy()
for x in range(0,len(keypoints1)):
  img=cv2.circle(img, (np.int(keypoints1[x].pt[0]),np.int(keypoints1[x].pt[1])), radius=np.int(keypoints1[x].size), color=(255), thickness=-1)
plt.axis("off")
plt.imshow(img)
plt.show()

# Detect blobs.
keypoints2 = detector.detect(255-im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints2, np.array([]), (0,0,255), 
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)
#plt.imshow(im_with_keypoints, cmap="gray")
#plt.show()

img = im.copy()
print("keypoints2 - Rings",len(keypoints2))   
for x in range(0,len(keypoints2)):
  img=cv2.circle(img, (np.int(keypoints2[x].pt[0]),np.int(keypoints2[x].pt[1])), radius=np.int(keypoints2[x].size), color=(255), thickness=-1)
plt.axis("off")
plt.imshow(img)
plt.show()


#print("keypoints1 - Points" ,len(keypoints1) )
#for keyPoint in keypoints1:
#    x = keyPoint.pt[0]
#    y = keyPoint.pt[1]
#    print("x , y ",(x,y))
#
#print("keypoints1 - Rings",len(keypoints2))    
#for keyPoint in keypoints2:
#    x = keyPoint.pt[0]
#    y = keyPoint.pt[1]
#    print("x , y ",(x,y))    
    
    
keypoints=[]
    
for keyPoint in keypoints1:
    keypoints.append(keyPoint)
    
for keyPoint in keypoints2:
    keypoints.append(keyPoint)    
    
    


print("keypoints - Centers",len(keypoints))    
for keyPoint in keypoints:
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
#    print("x , y ",(x,y))      

    

#im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


img = im.copy()
for x in range(0,len(keypoints)):
  img=cv2.circle(img, (np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1])), radius=np.int(keypoints[x].size), color=(255), thickness=-1)
plt.axis("off")
plt.imshow(img)
plt.show()

import csv

with open(r'C:\SAI\IIIT\2019_Spring\Assignment1\Assignment1_Data\r_up.csv', mode='w') as image_points_file:
    image_points_writer = csv.writer(image_points_file, delimiter=',')
    for keyPoint in keypoints:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        image_points_writer.writerow([x,y])


#h,w=im1.shape
#new_im = cv2.resize(img, (w,h)) 
#cv2.imwrite('a100_corners.jpg',new_im)

#https://stackoverflow.com/questions/42203898/python-opencv-blob-detection-or-circle-detection    


    