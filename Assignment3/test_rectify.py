import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy.linalg as la

# import os
# import time

_magic = [0.299, 0.587, 0.114]
_zero = [0, 0, 0]
_ident = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

true_anaglyph = ([_magic, _zero, _zero], [_zero, _zero, _magic])
gray_anaglyph = ([_magic, _zero, _zero], [_zero, _magic, _magic])
color_anaglyph = ([_ident[0], _zero, _zero], [_zero, _ident[1], _ident[2]])
half_color_anaglyph = ([_magic, _zero, _zero], [_zero, _ident[1], _ident[2]])
optimized_anaglyph = ([[0, 0.7, 0.3], _zero, _zero], [_zero, _ident[1], _ident[2]])
methods = [true_anaglyph, gray_anaglyph, color_anaglyph, half_color_anaglyph, optimized_anaglyph]

# SUBFUNCTIONS
# ============

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    
    if len(img1.shape) == 2: # grayscale input
        print( "Grayscale Input (drawLines)")
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)        
    elif len(img1.shape) == 3: # RGB
        print( "Color Input (drawLines)")
        
    else:
        print( len(img1.shape))
       
    r,c,ch = img1.shape 
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        
    return img1,img2

def findBestMatches(kp1,kp2,des1,des2):
    
    # THIS: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # https://gist.github.com/moshekaplan/5106221

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    pts1 = []
    pts2 = []
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    # Now we have the list of best matches from both the images
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2, good

def drawEpilinesAndMatches(imgL, imgR, pts1, pts2, kp1, kp2, good):

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img3_L = imgL
    img3_R = imgR
    img3_L, img3_R = drawlines(img3_L,img3_R,lines1,pts1,pts2)
    
    fig = plt.figure()    
    plt.subplot(121)
    plt.imshow(imgL), plt.title('Input L (no lines should be written on these variables?)')
    plt.subplot(122)
    plt.imshow(imgR), plt.title('Input R (no lines should be written on these variables?)')
    plt.show()    
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img4_L = imgL
    img4_R = imgR
    img4_L,img4_R = drawlines(img4_L,img4_R,lines2,pts2,pts1)

    
    # cv2.drawMatchesKnn expects list of lists as matches.
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    imgDummy = np.zeros((1,1))
    img5 = cv2.drawMatches(imgL,kp1,imgR,kp2,good[:10],imgDummy)
    
    return img3_L, img3_R, img4_L, img4_R, img5, lines1, lines2

def to_homg(x):
    """
    Transform x to homogeneous coordinates
    If X is MxN, returns an (M+1)xN array with ones on the last row
    >>> to_homg(np.array([[1, 2, 3], [1, 2, 3]], dtype=float))
    array([[ 1.,  2.,  3.],
           [ 1.,  2.,  3.],
           [ 1.,  1.,  1.]])
    >>> to_homg(np.array([[1], [2]]))
    array([[ 1.],
           [ 2.],
           [ 1.]])
    >>> to_homg([1, 2])
    array([1, 2, 1])
    """
    if hasattr(x, 'shape') and len(x.shape) > 1:
        return np.r_[x, np.ones((1,x.shape[1]))]
    else:
        return np.r_[x, 1]


def from_homg(x):
    """
    Transform homogeneous x to non-homogeneous coordinates
    If X is MxN, returns an (M-1)xN array that will contain nan when for
    columns where the last row was 0
    >>> from_homg(np.array([[1, 2, 3],
    ...                     [4, 5, 0]], dtype=float))
    array([[ 0.25,  0.4 ,   nan]])
    >>> from_homg(np.array([1, 5], dtype=float))
    array([ 0.2])
    >>> from_homg([1, 5, 0])
    array([ nan,  nan])
    >>> from_homg((1, 4, 0.5))end = time.time()
    array([ 2.,  8.])
    """
    if hasattr(x, 'shape') and len(x.shape) > 1:
        #valid = np.nonzero(x[-1,:])
        valid = x[-1,:] != 0
        result = np.empty((x.shape[0]-1, x.shape[1]), dtype=float)
        result[:,valid] = x[:-1,valid] / x[-1, valid]
        result[:,~valid] = np.nan
        return result
    else:
        if x[-1] == 0:
            result = np.empty(len(x)-1, dtype=float)
            result[:] = np.nan
            return result
        else:
            return np.array(x[:-1]) / x[-1]


def rectify_shearing(H1, H2, imsize):
    """Compute shearing transform than can be applied after the rectification
    transform to reduce distortion.
    See :
    http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
    "Computing rectifying homographies for stereo vision" by Loop & Zhang
    """
    w = imsize[0]
    h = imsize[1]

    a = ((w-1)/2., 0., 1.)
    b = (w-1., (h-1.)/2., 1.)
    c = ((w-1.)/2., h-1., 1.)
    d = (0., (h-1.)/2., 1.)

    ap = from_homg(H1.dot(a))
    bp = from_homg(H1.dot(b))
    cp = from_homg(H1.dot(c))
    dp = from_homg(H1.dot(d))

    x = bp - dp
    y = cp - ap

    k1 = (h*h*x[1]*x[1] + w*w*y[1]*y[1]) / (h*w*(x[1]*y[0] - x[0]*y[1]))
    k2 = (h*h*x[0]*x[1] + w*w*y[0]*y[1]) / (h*w*(x[0]*y[1] - x[1]*y[0]))

    if k1 < 0:
        k1 *= -1
        k2 *= -1

    return np.array([[k1, k2, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=float)

def rectify_uncalibrated(lines1, lines2, x1, x2, F, imsize, threshold=5):
    """
    Compute rectification homography for two images. This is based on
    algo 11.12.3 of HZ2
    This is also heavily inspired by cv::stereoRectifyUncalibrated
    Args:
        - imsize is (width, height)
    """
    U, W, V = la.svd(F)
    # Enforce rank 2 on fundamental matrix
    W[2] = 0
    W = np.diag(W)
    F = U.dot(W).dot(V)    

    # HZ2 11.12.1 : Compute H = GRT where :
    # - T is a translation taking point x0 to the origin
    # - R is a rotation about the origin taking the epipole e' to (f,0,1)
    # - G is a mapping taking (f,0,1) to infinity

    # e2 is the left null vector of F (the one corresponding to the singular
    # value that is 0 => the third column of U)
    e2 = U[:,2]

    # TODO: They do this in OpenCV, not sure why
    if e2[2] < 0:
        e2 *= -1

    # Translation bringing the image center to the origin
    # FIXME: This is kind of stupid, but to get the same results as OpenCV,
    # use cv.Round function, which has a strange behaviour :
    # cv.Round(99.5) => 100
    # cv.Round(132.5) => 132
    # cx = cv2.Round((imsize[0]-1)*0.5)
    cx = np.round((imsize[0]-1)*0.5)
    # cy = cv2.Round((imsize[1]-1)*0.5)
    cy = np.round((imsize[1]-1)*0.5)

    T = np.array([[1, 0, -cx],
                  [0, 1, -cy],
                  [0, 0, 1]], dtype=float)

    e2 = T.dot(e2)
    mirror = e2[0] < 0

    # Compute rotation matrix R that should bring e2 to (f,0,1)
    # 2D norm of the epipole, avoid division by zero
    d = max(np.sqrt(e2[0]*e2[0] + e2[1]*e2[1]), 1e-7)
    alpha = e2[0]/d
    beta = e2[1]/d
    R = np.array([[alpha, beta, 0],
                  [-beta, alpha, 0],
                  [0, 0, 1]], dtype=float)

    e2 = R.dot(e2)

    # Compute G : mapping taking (f,0,1) to infinity
    invf = 0 if abs(e2[2]) < 1e-6*abs(e2[0]) else -e2[2]/e2[0]
    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [invf, 0, 1]], dtype=float)

    # Map the origin back to the center of the image
    iT = np.array([[1, 0, cx],
                   [0, 1, cy],
                   [0, 0, 1]], dtype=float)

    H2 = iT.dot(G.dot(R.dot(T)))

    # HZ2 11.12.2 : Find matching projective transform H1 that minimize
    # least-square distance between reprojected points
    e2 = U[:,2]

    # TODO: They do this in OpenCV, not sure why
    if e2[2] < 0:
        e2 *= -1

    e2_x = np.array([[0, -e2[2], e2[1]],
                     [e2[2], 0, -e2[0]],
                     [-e2[1], e2[0], 0]], dtype=float)

    e2_111 = np.array([[e2[0], e2[0], e2[0]],
                       [e2[1], e2[1], e2[1]],
                       [e2[2], e2[2], e2[2]]], dtype=float)

    H0 = H2.dot(e2_x.dot(F) + e2_111)

    # Minimize \sum{(a*x_i + b*y_i + c - x'_i)^2} (HZ2 p.307)
    # Compute H1*x1 and H2*x2    
    x1h = to_homg(x1)
    x2h = to_homg(x2)
    try:
        A = H0.dot(x1h).T
    except ValueError:
        print( "Error, TODO: Why this kind of dimensions (H0*x1), need to transpose pts1?")
        print( " H0 dim: " + str(H0.shape))
        print( "  x2h dim: " + str(x1h.shape))
        A = H0 # nothing done

    # We want last (homogeneous) coordinate to be 1 (coefficient of c
    # in the equation)
    A = (A.T / A[:,2]).T # for some reason, A / A[:,2] doesn't work
    try:
        B = H2.dot(x2h)
    except ValueError:
        print( "Error, TODO: Why this kind of dimensions (H2*x2), need to transpose pts2?")
        print( " H2 dim: " + str(H2.shape))
        print( "  x1h dim: " + str(x1h.shape))
        B = H2 # nothing done
        
    B = B / B[2,:] # from homogeneous
    B = B[0,:] # only interested in x coordinate

    X, _, _, _ = la.lstsq(A, B)

    # Build Ha (HZ2 eq. 11.20)
    Ha = np.array([[X[0], X[1], X[2]],
                   [0, 1, 0],
                   [0, 0, 1]], dtype=float)

    H1 = Ha.dot(H0)

    if mirror:
        mm = np.array([[-1, 0, cx*2],
                       [0, -1, cy*2],
                       [0, 0, 1]], dtype=float)
        H1 = mm.dot(H1)
        H2 = mm.dot(H2)

    return H1, H2
  
def undistortMap(H1, H2, img1, img2):
    
    # http://stackoverflow.com/questions/10192552/rectification-of-uncalibrated-cameras-via-fundamental-matrix
    # http://ece631web.groups.et.byu.net/Lectures/ECEn631%2014%20-%20Calibration%20and%20Rectification.pdf
   
    imgsize = (imgL.shape[1], imgL.shape[0])
    
    K = np.array([[50, 0, 20], [0, 50, 30], [0, 0, 1]]) # guess by PT    
    d = None # Assume no distortion
   
    rH = la.inv(K).dot(H1).dot(K)
    lH = la.inv(K).dot(H2).dot(K)

    

    # TODO: lRect or rRect for img1/img2 ??
    map1x, map1y = cv2.initUndistortRectifyMap(K, d, rH, K, imgsize,
                                               cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K, d, lH, K, imgsize,
                                               cv2.CV_16SC2)

    # Convert the images to RGBA (add an axis with 4 values)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2RGBA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2RGBA)
    
    return img1, map1x, map1y, img2, map2x, map2y


def remap(img1, map1x, map1y, img2, map2x, map2y):

    rimg1 = cv2.remap(img1, map1x, map1y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0,0,0,0))
    rimg2 = cv2.remap(img2, map2x, map2y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0,0,0,0))

    # Put a red background on the invalid values
    # TODO: Return a mask for valid/invalid values
    # TODO: There is aliasing hapenning on the images border. We should
    # invalidate a margin around the border so we're sure we have only valid
    # pixels
    '''
    if len(img1.shape) == 2: # grayscale
        # print "Grayscale for remap"
        rimg1[rimg1[:,:,3] == 0,:] = (255,0,0,255)
        rimg2[rimg2[:,:,3] == 0,:] = (255,0,0,255)
    elif len(img1.shape) == 3: # RGB
        # print "Color for remap"
        rimg1[rimg1[:,:,3] == 0,:] = (255,0,0,255)
        rimg2[rimg2[:,:,3] == 0,:] = (255,0,0,255)
    elif len(img1.shape) == 4: # RGBA
        # print "Color (RGBA) for remap"
        rimg1[rimg1[:,:,3] == 0,:] = (255,0,0,255)
        rimg2[rimg2[:,:,3] == 0,:] = (255,0,0,255)
    else:
        print str(len(img1.shape)) + " image size / type (remap)?"
    '''
        
    return rimg1, rimg2
    

def show_rectified_images(rimg1, rimg2):
    
    plt.subplot(247)
    plt.imshow(rimg1)

    # Hack to get the lines span on the left image
    # http://stackoverflow.com/questions/6146290/plotting-a-line-over-several-graphs
    for i in xrange(1, rimg1.shape[0], rimg1.shape[0]/20):
        plt.axhline(y=i, color='g', xmin=0, xmax=1.2, clip_on=False);

    plt.subplot(248)
    plt.imshow(rimg2)
    for i in xrange(1, rimg1.shape[0], rimg1.shape[0]/20):
        plt.axhline(y=i, color='g');
        

def rectifyWrapper(imgL, imgR, lines1, lines2, pts1, pts2, F):
    
    # We need to apply the perspective transformation
    # e.g. http://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
        
    # Transform the images so the matching horizontal lines will be horizontal 
    # with each other between images, http://scientiatertiidimension.blogspot.ca/2013/11/playing-with-disparity.html
    # (hint: cv2.stereoRectifyUncalibrated, cv2.warpPerspective).
    imgsize = (imgL.shape[1], imgL.shape[0])
    
    # transpose
    pts1 = pts1.T
    pts2 = pts2.T
    
    try:
       _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgsize)
    except cv2.error:
        print ("cv2.error")
        H1, H2 = rectify_uncalibrated(lines1, lines2, pts1, pts2, F, imgsize)
        print ("applying the custom-rectify code then as the cv2.stereoRectifyUncalibrated failed")
        
    # http://stackoverflow.com/questions/19704369/stereorectifyuncalibrated-not-accepting-same-array-as-findfundamentalmat
    # OpenCV Error: Assertion failed (CV_IS_MAT(_points1) && CV_IS_MAT(_points2) && ... 
    
    # correct for shearing
    # http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
    S = rectify_shearing(H1, H2, imgsize)
    H1 = S.dot(H1)

    
    # Init Undistort, Map (mapx / mapy)
    img1, map1x, map1y, img2, map2x, map2y = undistortMap(H1, H2, imgL, imgR)
   
    # Remap
    rimg1, rimg2 = remap(img1, map1x, map1y, img2, map2x, map2y)
    
    return rimg1, rimg2

def disparityMapStereo(imgL,imgR):
    
    # BLOCK MATCHING ALGORITHM
    stereoBM = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparityBM = stereoBM.compute(imgL[:,:,2],imgR[:,:,2])
    
    return disparityBM


# http://bytes.com/topic/python/answers/486627-anaglyph-3d-stereo-imaging-pil-numpy
def anaglyph(image1, image2, method=true_anaglyph):
    
    m1, m2 = [np.array(m).transpose() for m in method]
    image1 = np.dot(image1, m1) # float64
    image2 = np.dot(image2, m2) # int64    
    composite = cv2.add(np.asarray(image1, dtype="uint8"), np.asarray(image2, dtype="uint8"))    
    return composite

def segmentImages(tr_img1, tr_img2):

    return tr_img1, tr_img2

def registerImages(src_pts, dst_pts, src_img, dst_img):

    # HOMOGRAPHY
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)    
    height, width, channels = dst_img.shape    
    out = cv2.warpPerspective(dst_img[:,:,2], M, (width, height))
    out2 = cv2.warpPerspective(src_img[:,:,2], M, (width, height))
    
    # PERSPECTIVE TRANSFORM
    '''
    M2 = cv2.getPerspectiveTransform(np.asarray(src_pts, dtype="float32"), np.asarray(dst_pts, dtype="float32"))
    out3 = cv2.warpPerspective(dst_img[:,:,2], M2, (width, height))
    out4 = cv2.warpPerspective(src_img[:,:,2], M2, (width, height))
    '''
    
    fig = plt.figure()
    
    # HOMOGRAPHY
    ax1 = plt.subplot2grid((2,5), (0,0))
    plt.imshow(src_img), plt.title('Source')
    ax2 = plt.subplot2grid((2,5), (0,1))
    plt.imshow(dst_img), plt.title('Destination')
    ax3 = plt.subplot2grid((2,5), (0,2))
    plt.imshow(out), plt.title('Homo Out (dst)')
    ax4 = plt.subplot2grid((2,5), (0,3))
    plt.imshow(out2), plt.title('Homo Out (src)')
    ax5 = plt.subplot2grid((2,5), (0,4))
    plt.imshow(cv2.addWeighted(out, 0.5, out2, 0.5, 0)), plt.title('Overlay')
    
    '''
    # PERSPECTIVE TRANSFORM
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    ax1b = plt.subplot2grid((2,5), (1,0))
    plt.imshow(src_img), plt.title('Source')
    ax2b = plt.subplot2grid((2,5), (1,1))
    plt.imshow(dst_img), plt.title('Destination')
    ax3b = plt.subplot2grid((2,5), (1,2))
    plt.imshow(out3), plt.title('Persp Out (dst)')
    ax4b = plt.subplot2grid((2,5), (1,3))
    plt.imshow(out4), plt.title('Persp Out (src)')
    ax5b = plt.subplot2grid((2,5), (1,4))
    plt.imshow(cv2.addWeighted(out3, 0.5, out4, 0.5, 0)), plt.title('Overlay')
    '''
    
    plt.show()                   
    return out
    

def getColorSpaces(image):  
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  
    
    return rgb,gray

# MAIN CODE
# ============

if __name__ == "__main__":

    # Read images as grayscales
#    imgL = cv2.imread('testImages/pt_20150621_stereoTest_001left.jpg',1)
#    imgR = cv2.imread('testImages/pt_20150621_stereoTest_002right.jpg',1)
#    
    img = cv2.imread(r'Stereo Images\Stereo_Pair1.jpg')
    

    height,width = img.shape[:2]
    width_cutoff = width // 2
    imgL = img[:, :width_cutoff]
    imgR = img[:, width_cutoff:]

#    rgbL,imgL=getColorSpaces(imgL)
#    rgbR,imgR=getColorSpaces(imgR)
    
    
    # From "Middlebury Stereo Dataset" | Tsukuba:
    # http://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/
    # imgL = cv2.imread('testImages/scene1.row3.col2.ppm')
    # imgR = cv2.imread('testImages/scene1.row3.col4.ppm')      
    
    # Easier to work with RGB order rather than with the BGR of openCV
    # http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_matplotlib_rgb_brg_image_load_display_save.php
    if len(imgL.shape) == 2: # Greyscale
        imgL = cv2.cvtColor(imgL,cv2.COLOR_GRAY2RGB)
        imgR = cv2.cvtColor(imgR,cv2.COLOR_GRAY2RGB)
    if len(imgL.shape) == 3: # RGB
        imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB)

    # Initiate SURF detector
    surf_threshold = 6000
    surf = cv2.xfeatures2d.SURF_create(surf_threshold)
    
    # Find KEYPOINTS and DESCRIPTORS
    
    # For some reason it was more robust using the grayscale as the input image for this
    # even though in theory at least it should be easier to use the color information as well 
    # along the luminosity to find the keypoints/descriptors
    surfImg_L = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    surfImg_R = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
    
    kp1, des1 = surf.detectAndCompute(surfImg_L,None)
    kp2, des2 = surf.detectAndCompute(surfImg_R,None)
    
    # Now we can match the keypoints
    pts1, pts2, good = findBestMatches(kp1,kp2,des1,des2)
    
    # Compute the Fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    
    # We select only inlier points based on fundamental matrix
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    # Draw the results of this process    
    # img3_L, img3_R, img4_L, img4_R, img5, lines1, lines2 = drawEpilinesAndMatches(imgL, imgR, pts1, pts2, kp1, kp2, good)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    
    print(len(lines1))
    print(len(lines2))
    # Rectify the images now (Find homography matrices, undistort, remap)
    rimg1, rimg2 = rectifyWrapper(imgL, imgR, lines1, lines2, pts1, pts2, F)
    
    # Overlay (align, register) the images
    overlayIN = cv2.addWeighted(imgL, 0.5, imgR, 0.5, 0)
    overlayRect = cv2.addWeighted(rimg1, 0.5, rimg2, 0.5, 0)
    
    # Register & Warp
    # tr_img2 = registerImages(pts1, pts2, rimg1[:,:,0:3], rimg2[:,:,0:3])        
    
    # Compute disparity map
    # disparityBM = disparityMapStereo(rimg1, rimg2)
    
    # Segment the image to foreground/background, so that the anaglyph composite
    # is only created from the foreground
    foreground, background = segmentImages(rimg1, rimg2)
    
    # Create the anaglyph
    anaglyph = anaglyph(rimg1[:,:,0:3], rimg2[:,:,0:3], optimized_anaglyph)
    # anaglyph = anaglyph(rimg1[:,:,0:3], tr_img2[:,:,0:3], optimized_anaglyph)
    # anaglyph = anaglyphWithForeground(rimg1[:,:,0:3], rimg2[:,:,0:3], optimized_anaglyph, foreground, background)
        
    # DISPLAY
    # ============
    
    print( "Keypoints from LEFT: " + str(len(kp1))) # 943 at threshold = 400; 44 at threshold=6000, 93 at thre = 4000
    print( "Keypoints from RIGHT: " + str(len(kp2))) # 943 at threshold = 400; 44 at threshold=6000, 102 at thre = 4000
    
    print( "NOT WORKING YET:")
    # print "H2*x2, and H0*x1 not working" # fixed by transposing pts1, and pts2
    print( "Fix the Camera matrix K")
    print( ".. no distortion at the moment" )# not needed really for 50mm/f1.8, but still
    
    # cv2.drawMatchesKnn expects list of lists as matches.
    # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    imgDummy = np.zeros((1,1))
    img5 = cv2.drawMatches(imgL,kp1,imgR,kp2,good[:10],imgDummy)
    
    # Draw the points for display 
    img2_L = cv2.drawKeypoints(imgL,kp1,None,(255,0,0),4)
    img2_R = cv2.drawKeypoints(imgR,kp2,None,(255,0,0),4)
    
    # PLOT 1
    # subplot customization: subplot2grid
    # http://matplotlib.org/users/gridspec.html
    plt.close('all') # will close all figures


    fig = plt.figure(facecolor='white',figsize=(12,12))
    #fig.set_figheight(11)
    #fig.set_figwidth(8.5)
    
    # subplot layout
    rows = 3
    cols = 3
    
    ax1 = plt.subplot2grid((rows,cols), (0,0)), plt.axis('off')
    plt.imshow(overlayIN), plt.title('Input Overlay')
    
    ax4 = plt.subplot2grid((rows,cols), (1,1), colspan=2, rowspan=2)
    plt.imshow(anaglyph), plt.title('Anaglyph'), plt.axis('off')
    
    ax5 = plt.subplot2grid((rows,cols), (1,0))
    plt.imshow(rimg1), plt.title('Rectified Left'), plt.axis('off')
    ax6 = plt.subplot2grid((rows,cols), (2,0))
    plt.imshow(rimg2), plt.title('Rectified Right'), plt.axis('off')
    
        
    ax2 = plt.subplot2grid((rows,cols), (0,1), colspan=2)
    plt.imshow(img5), plt.title('Matching Features'), plt.axis('off')
        
    plt.show()
    
#    for i in range(3,27,2):
        
#    stereo = cv2.StereoSGBM_create(minDisparity=16, numDisparities=16*16,blockSize=7)  
#    L = np.abs(rimg1)
#    R = np.abs(rimg2)
#    L = cv2.normalize(L, L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#    R = cv2.normalize(R, R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#    disparity = stereo.compute(L,R)
##    disparity = stereo.compute(rimg1,rimg2)
#    plt.axis('off')
#    plt.title(i)
#    plt.imshow(disparity,'gray')
#    plt.show()