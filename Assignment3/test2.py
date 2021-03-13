import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def getColorSpaces(image):  
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  
    
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

def saveImage(title, image):
    image = np.divide(image, image.max())
    cv2.imwrite(title+str(random.randint(1, 100))+'.jpg', image*255)

def getDTWDisparity(grayL,grayR,occ_cost):
    
    height,width = grayL.shape[:2]

    dispL = np.zeros(grayL.shape[:2])
    dispR = np.zeros(grayR.shape[:2])

    for x in range(height):
        cost_Mat = np.zeros((width+1, width+1))
        dir_Mat = np.zeros((width+1, width+1))

        for i in range(1, width+1):
            cost_Mat[i, 0] = i * occ_cost
            cost_Mat[0, i] = i * occ_cost

        for r in range(1, width+1):
            for c in range(1, width+1):
                min1 = cost_Mat[r-1, c-1] + np.abs(grayL[x, r-1] - grayR[x, c-1])
                min2 = cost_Mat[r-1, c] + occ_cost
                min3 = cost_Mat[r, c-1] + occ_cost

                cost_Mat[r, c] = min([min1, min2, min3])
                cmin = cost_Mat[r, c]

                if cmin == min1 :
                    dir_Mat[r, c] = 1
                elif cmin == min2 :
                    dir_Mat[r, c] = 2
                elif cmin == min3 :
                    dir_Mat[r, c] = 3

        p = width
        q = width

        while (p != 0 and q != 0):
            print(".", end="", flush=True)
            if dir_Mat[p, q] == 1 :
                p = p - 1
                q = q - 1
                dispL[x][p] = np.abs(p - q)
                dispR[x][q] = np.abs(p - q)
            elif dir_Mat[p][q] == 2 :
                p = p - 1
                dispL[x][p] = np.abs(p - q)
            elif dir_Mat[p][q] == 3 :
                q = q - 1
                dispR[x][q] = np.abs(p - q)
                
    return   dispL,dispR          
# Read the images
img = cv2.imread(r'Stereo Images\Stereo_Pair1.jpg')

height,width = img.shape[:2]
width_cutoff = width // 2
imgL = img[:, :width_cutoff]
imgR = img[:, width_cutoff:]

imgL=cv2.pyrDown(imgL)
imgR=cv2.pyrDown(imgR)
imgL=cv2.pyrDown(imgL)
imgR=cv2.pyrDown(imgR)


#imgL=cv2.imread(r'Stereo Images\tsucuba_left.png')
#imgR=cv2.imread(r'Stereo Images\tsucuba_right.png')
rgbL,grayL=getColorSpaces(imgL)
rgbR,grayR=getColorSpaces(imgR)

#Display the images
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
showImage(rgbL,'Left Image')
plt.subplot(1,2,2)
showImage(rgbR,'Right Image')
plt.show()

occ_cost = 15

dispL,dispR=getDTWDisparity(grayL,grayR,occ_cost)
saveImage('Left_', dispL)
saveImage('Right_', dispR)




