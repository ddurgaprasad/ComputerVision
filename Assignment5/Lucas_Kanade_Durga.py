import numpy as np
import cv2
import os
from skimage import filters
from scipy.ndimage.filters import convolve
from matplotlib import pyplot as plt

def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    return rgb,gray

def getImageDimnesion(image):
    height,width = image.shape[:2]

    return height,width

def showImage(image,title,cmap):
    plt.imshow(image,cmap=cmap)
    plt.axis('off')
    plt.title(title)

def Lucas_Kanade(img1, img2):

    color = np.random.randint(0, 255, (100, 3))
    
    # parameter to get features
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    features= cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)  #using opencv function to get feature for which we are plotting flow
    feature = np.int32(features)
    # print(feature)
    feature = np.reshape(feature, newshape=[-1, 2])

    status=np.zeros(feature.shape[0]) # this will tell change in x,y

    mask = np.zeros_like(img2)

    newFeature=np.zeros_like(feature)
    
    for a,i in enumerate(feature):

        x, y = i


        newFeature[a]=[np.int32(x+u[y,x]),np.int32(y+v[y,x])]
        if np.int32(x+u[y,x])==x and np.int32(y+v[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
            status[a]=0
        else:
            status[a]=1 # this tells us that x+dx , y+dy is not equal to x and y


    good_new=newFeature[status==1] #status will tell the position where x and y are changed so for plotting getting only that points
    good_old = feature[status==1]
    print(good_new.shape)
    print(good_old.shape)

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        img2 = cv2.circle(img2, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(img2, mask)
    return img


def lucas_kandae_optical_flow(Img1, Img2, window_size=3, tau=1e-2):

    height,width=getImageDimnesion(Img1)
    kernel_t=np.ones([2,2])

    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    Img1 = Img1 / 255. # normalize pixels
    Img2 = Img2 / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t

    #Page 20 , Mubarak Shah presentation
    fx = filters.sobel_v(Img1)
    fy = filters.sobel_h(Img1)
    ft = convolve(Img2, kernel_t) + convolve(Img1, -kernel_t)

    #Motion vector
    #Image Displacement in x and y directions between two consecutive frames
    u = np.zeros(Img1.shape)
    v = np.zeros(Img1.shape)

    # within window window_size * window_size
    for i in range(w, Img1.shape[0]-w):
        for j in range(w, Img1.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            Ix = Ix[:]
            Iy = Iy[:]
            b = -It[:]

            A = np.vstack((Ix, Iy)).T
            #Page 30 Pseudo Inverse
            nu = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A)), A.T), b)

            u[i,j]=nu[0]
            v[i,j]=nu[1]

    return (u,v)


images_path=r'C:\SAI\IIIT\2019_Spring\Assignment5\eval-gray-twoframes\eval-data-gray'

frame1=[]
frame2=[]

for imgpair in os.listdir(images_path):
    imgpair_path=os.path.join(images_path,imgpair)
    frame1_path=os.path.join(imgpair_path,'frame10.png')
    frame2_path=os.path.join(imgpair_path,'frame11.png')
    frame1.append(frame1_path)
    frame2.append(frame2_path)

num_pairs=len(os.listdir(images_path))
pair_name=os.listdir(images_path)

for pair in range(1):

    ORIGINAL_IMAGE_1=cv2.imread(frame1[pair]);
    ORIGINAL_IMAGE_2=cv2.imread(frame2[pair]);

    rgb1,gray1=getColorSpaces(ORIGINAL_IMAGE_1)
    rgb2,gray2=getColorSpaces(ORIGINAL_IMAGE_2)

    I1_smooth = cv2.GaussianBlur(gray1, (3,3), 0)
    I2_smooth = cv2.GaussianBlur(gray2, (3,3), 0)

    # finding the good features
    features = cv2.goodFeaturesToTrack(I1_smooth,10000 ,0.01 ,10)
    
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.title('Frame 1')

    plt.imshow(I1_smooth, cmap ='gray')
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.title('Frame 2')

    plt.imshow(I2_smooth, cmap='gray')
    feature = np.int0(features)
    for i in feature:
        x,y = i.ravel()
        cv2.circle(I1_smooth,(x,y) ,3 	,0 ,-1 )

    plt.show()

    u,v=lucas_kandae_optical_flow(gray1,gray2)

    n, m = u.shape
    u_deci = u[np.ix_(range(0, n, 5), range(0, m, 5))]
    v_deci = v[np.ix_(range(0, n, 5), range(0, m, 5))]
    [X,Y] = np.meshgrid(np.arange(m, dtype = 'float64'), np.arange(n, dtype = 'float64'))
    X_deci = X[np.ix_(range(0, n, 5), range(0, m, 5))]
    Y_deci = Y[np.ix_(range(0, n, 5), range(0, m, 5))]

    plt.title(pair_name[pair])
    plt.axis('off')
    plt.imshow(rgb1, cmap = 'gray')
    plt.quiver(X_deci, Y_deci, u_deci, v_deci, color='b')
    plt.show()


##https://github.com/tnybny/Exploring-optical-flow/blob/master/Lucas-Kanade%20optical%20flow%20example.ipynb
#







