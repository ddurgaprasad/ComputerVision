# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 08:40:35 2019

@author: E442282
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def getColorSpaces(image):  
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)  
    
    return rgb,gray
    
def getImageDimnesion(image):    
    height,width = image.shape[:2]
    
    return height,width

def showImage(image,title):    
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.title(title)
      
def getSubImage(img,left,right,top,bottom):
    
    return img[top:bottom, left:right].astype(np.float32)


#img = cv2.imread(r'Stereo Images\Stereo_Pair1.jpg')
#
#height,width = img.shape[:2]
#width_cutoff = width // 2
#imgL = img[:, :width_cutoff]
#imgR = img[:, width_cutoff:]
#
#imgL=cv2.pyrDown(imgL)
#imgR=cv2.pyrDown(imgR)
#imgL=cv2.pyrDown(imgL)
#imgR=cv2.pyrDown(imgR)


imgL=cv2.imread(r'Stereo Images\tsucuba_left.png')
imgR=cv2.imread(r'Stereo Images\tsucuba_right.png')
rgbL,grayL=getColorSpaces(imgL)
rgbR,grayR=getColorSpaces(imgR)

#Display the images
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
showImage(rgbL,'Left Image')
plt.subplot(1,2,2)
showImage(rgbR,'Right Image')
plt.show()


euclidean_dist = lambda x, y: np.abs(x - y)
# Distances
def getDistances(y,x):
    distances = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            distances[i,j] = euclidean_dist(x[j],y[i])
            
    return distances         

def distance_cost_plot(distances):
    plt.imshow(distances, interpolation='nearest', cmap='Reds') 
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.colorbar();
    
# Accumulated Cost
def getAccumulatedCost(x,y,distances):
    
    accumulated_cost = np.zeros((len(y), len(x)))
    accumulated_cost[0,0] = distances[0,0]
   
    #First row
    for i in range(1, len(x)):
        accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]
    #First column
    for i in range(1, len(y)):
        accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0]   
    
    #Rest of the columns/rows
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]

    return accumulated_cost

def getBacktrackedCost(x, y, accumulated_cost, distances):
    path = [[len(x)-1, len(y)-1]]
    cost = 0
    i = len(y)-1
    j = len(x)-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j= j- 1
        path.append([j, i])
    path.append([0,0])
    for [y, x] in path:
        cost = cost +distances[x, y]
        
        
    return path, cost    

def getDisparity(rowL,rowR):
    dist_matrix=getDistances(rowL,rowR)    
    acc_cost_matrix=getAccumulatedCost(rowL,rowR,dist_matrix)
    path, cost = getBacktrackedCost(rowL, rowR, acc_cost_matrix, dist_matrix)
#    path_x = [point[0] for point in path]
#    path_y = [point[1] for point in path]  
    
#    disparity=path_x-path_y
    disparity=[point[0]-point[1] for point in path]
#    distance_cost_plot(acc_cost_matrix)
    
    
#    plt.plot(path_x, path_y);
#    print("path_x len " ,len(path_x))
#    print("path_y len " ,len(path_y))
#    
#    print("path_x  " ,path_x)  
#    print("path_y  " ,path_y)    
#    print("disparity  " ,disparity)
    
    return disparity
    

rowL=grayL[0]
rowR=grayR[0]

disparity=getDisparity(rowL,rowR)    
print("disparity " ,len(disparity))  

   
## First row from Left and Right images
#rowL=grayL[0]
#rowR=grayR[0]
#
#disparity=getDisparity(rowL,rowR)
#print(len(disparity))

height,width=getImageDimnesion(grayL)


print("Image Dimension " ,height,width)


#disparity=disparity[:width]


disp_map = np.empty((0,width), int)

for i in range(height):
    print(".", end="", flush=True)
    rowL=grayL[i]
    rowR=grayR[i]
    disparity=getDisparity(rowL,rowR)  
    
    disparity=disparity[:width]
#    disp_map = np.vstack([disp_map, disparity])
    disp_map=np.append(disp_map, np.array([disparity]), axis=0)

 


























