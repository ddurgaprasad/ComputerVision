# ps2

import numpy as np
import cv2
import matplotlib.pyplot as plt

def getColorSpaces(image):  
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  
    
    return rgb,gray
    
def getImageDimnesion(image):    
    height,width = image.shape[:2]
    
    return height,width

def showImage(image,title,row_col,fig_width,fig_height):    
    plt.figure(figsize=(fig_width,fig_height))
    plt.subplot(row_col[0],row_col[1],row_col[2])
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.title(title)
      

def getSubImage(img,left,right,top,bottom):
    
    return img[top:bottom, left:right].astype(np.float32)

def getMaxNCC(row_dest, template):
    res = cv2.matchTemplate(row_dest, template, cv2.TM_CCOEFF_NORMED)
    max_sim = np.argmax(res)
    return max_sim


def getNCC(windowL,windowR):
    
    #A solution is to NORMALIZE the pixels in the windows 
    #before comparing them by subtracting the mean of the of the
    #patch intensities and dividing by the std.dev std.dev.  
    windowL_Mean_Substract=windowL - np.mean(windowL)
    windowL_Standard_deviation= np.sqrt(np.sum(np.square(windowL_Mean_Substract)))
    
    windowR_Mean_Substract=windowL - np.mean(windowR)
    windowR_Standard_deviation= np.sqrt(np.sum(np.square(windowR_Mean_Substract)))
    
    numerator= windowL_Mean_Substract*windowR_Mean_Substract
    denominator=windowL_Standard_deviation*windowR_Standard_deviation
    
    ncc=np.sum(numerator/denominator)
    
    return ncc

def getMatchingLocation(template,Right_strip):
    
    template_H,template_W=template.shape[:2]
    strip_H,strip_W=Right_strip.shape[:2]

    corr_list=[]

    for i in range (0,strip_W-template_W):
        
        right_block=getSubImage(Right_strip,i,i+template_W,0,strip_H)
        corr=getNCC(template,right_block)

        corr_list.append((corr,i))
    sorted_corr=sorted(corr_list,key=lambda x:x[1],reverse=True)
    
    
    return sorted_corr[0][1]


matchPixels=[]

#https://github.com/gkouros/intro-to-cv-ud810/blob/master/ps2_python/disparity_ssd.py
def getDisparity_NCC(LeftImage,RightImage,block_size=7,disparity_range=30):

    height,width= getImageDimnesion(LeftImage)
    
    disparity_image = np.zeros(LeftImage.shape, dtype=np.float32)

    halfBlockSize=int(block_size/2) 


    for row in range(halfBlockSize, height-halfBlockSize):
        print(".", end="", flush=True) 
        template_row_min  = int(max(row-halfBlockSize, 0)) 
        template_row_max =  int(min(row+halfBlockSize+1, height))
        
        for col in range(halfBlockSize, width-halfBlockSize):
#            best_offset = 0
#            max_ncc = 0.0
            # get template
            template_col_min = int(max(col-halfBlockSize, 0))
            template_col_max = int(min(col+halfBlockSize+1, width))
            template =  getSubImage(LeftImage,template_col_min,template_col_max,template_row_min,template_row_max)

            # get row strip in a window with width=disparity_range
            strip_col_min = int(max(col-disparity_range/2, 0))
            strip_col_max = int(min(col+disparity_range/2+1, width))
            
            Right_strip = getSubImage(RightImage,strip_col_min,strip_col_max,template_row_min,template_row_max)
#            strip_H,strip_W=Right_strip.shape[:2]
#            template_H,template_W=template.shape[:2]
#            
#            for offset in range (0,strip_W-template_W):
#                right_block=getSubImage(Right_strip,offset,offset+template_W,0,strip_H)                 
#                ncc=getMaxNCC(template,right_block)                
#                if (max_ncc < ncc):
#                    max_ncc = ncc
#                    best_offset = offset
#                
            
            res = cv2.matchTemplate(Right_strip, template, method=cv2.TM_CCORR_NORMED) 
            template_col = int(max(col-strip_col_min-halfBlockSize, 0))
            diff = np.arange(res.shape[1]) - template_col
            match_loc =np.argmax(res)
            disparity_image[row, col] = diff[match_loc]
#            disparity_image[row, col] = best_offset
            matchPixels.append([row,col,row,match_loc])

    return disparity_image  


# Read the images
img = cv2.imread(r'Stereo Images\Stereo_Pair1.jpg')

height,width = img.shape[:2]
width_cutoff = width // 2
imgL = img[:, :width_cutoff]
imgR = img[:, width_cutoff:]

#imgL=cv2.imread(r'Stereo Images\Stereo_Pair1_Left.png')
#imgR=cv2.imread(r'Stereo Images\Stereo_Pair1_Right.png')
#

imgL=cv2.imread(r'Stereo Images\lion_left_rect.png')
imgR=cv2.imread(r'Stereo Images\lion_right_rect.png')



# Get RGB and Gray versions of the input images
# This Gray has a range of 0-255
rgbL,grayL=getColorSpaces(imgL)
rgbR,grayR=getColorSpaces(imgR)

#Display the images
showImage(rgbL,'Left Image',(1,2,1),12,12)
showImage(rgbR,'Right Image',(1,2,2),12,12)
plt.show()


grayL = grayL * (1.0 / 255.0)
grayR = grayR * (1.0 / 255.0)


#Filter window size
block_size=15

disparity_range=100  

disp = np.abs(getDisparity_NCC(grayL, grayR, block_size, disparity_range))
# shift and scale disparity maps
disp_norm = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#showImage(disp,'Disparity',(1,2,1),12,12)
showImage(disp_norm,'Disparity Normalized',(1,2,2),12,12)
plt.show()



