import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

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
    
    for i in range(1, len(x)):
        accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]
    for i in range(1, len(y)):
        accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0]   
      
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
    path_x = [point[0] for point in path]
    path_y = [point[1] for point in path]  
    
#    disparity=path_x-path_y
    disparity=[point[0]-point[1] for point in path]
    distance_cost_plot(acc_cost_matrix)
    plt.plot(path_x, path_y);
    print("path_x " ,len(path_x))
    print("path_y " ,len(path_y))
    
    return disparity
    

    

rowL = np.array([1, 1, 2, 3, 2, 0])
rowR = np.array([0, 1, 1, 2, 3, 2])

disparity=getDisparity(rowL,rowR)    
print("disparity " ,len(disparity))    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    