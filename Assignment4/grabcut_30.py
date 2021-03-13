# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:21:04 2019

@author: E442282
"""
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''


import numpy as np
import cv2 as cv
import igraph as ig
from GMM import GaussianMixture
import maxflow
from matplotlib import pyplot as plt
import cv2
import os

DRAW_BG = 0
DRAW_FG = 1
DRAW_PR_BG =2
DRAW_PR_FG = 3




class GrabCut:
    def __init__(self, img, mask, rect=None, gmm_components=5):
        self.img = np.asarray(img, dtype=np.float64)
        self.rows, self.cols, _ = img.shape

        self.mask = mask
        if rect is not None:
            self.mask[rect[1]:rect[1] + rect[3],
                      rect[0]:rect[0] + rect[2]] = DRAW_PR_FG
        self.classify_pixels()

        # Best number of GMM components K suggested in paper
        self.gmm_components = gmm_components
        self.gamma = 50  # Best gamma suggested in paper formula (5)
        self.beta = 0

        self.left_V = np.empty((self.rows, self.cols - 1))
        self.upleft_V = np.empty((self.rows - 1, self.cols - 1))
        self.up_V = np.empty((self.rows - 1, self.cols))
        self.upright_V = np.empty((self.rows - 1, self.cols - 1))

        self.bgd_gmm = None
        self.fgd_gmm = None
        self.comp_idxs = np.empty((self.rows, self.cols), dtype=np.uint32)

        self.gc_graph = None
        self.gc_graph_capacity = None           # Edge capacities
        self.gc_source = self.cols * self.rows  # "object" terminal S
        self.gc_sink = self.gc_source + 1       # "background" terminal T

        self.calc_beta_smoothness()
        self.init_GMMs()
        self.run()

    def calc_beta_smoothness(self):
        _left_diff = self.img[:, 1:] - self.img[:, :-1]
        _upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1]
        _up_diff = self.img[1:, :] - self.img[:-1, :]
        _upright_diff = self.img[1:, :-1] - self.img[:-1, 1:]

        self.beta = np.sum(np.square(_left_diff)) + np.sum(np.square(_upleft_diff)) + \
            np.sum(np.square(_up_diff)) + \
            np.sum(np.square(_upright_diff))
        self.beta = 1 / (2 * self.beta / (
            # Each pixel has 4 neighbors (left, upleft, up, upright)
            4 * self.cols * self.rows
            # The 1st column doesn't have left, upleft and the last column doesn't have upright
            - 3 * self.cols
            - 3 * self.rows  # The first row doesn't have upleft, up and upright
            + 2))  # The first and last pixels in the 1st row are removed twice
        print('Beta:', self.beta)

        # Smoothness term V described in formula (11)
        self.left_V = self.gamma * np.exp(-self.beta * np.sum(
            np.square(_left_diff), axis=2))
        self.upleft_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(
            np.square(_upleft_diff), axis=2))
        self.up_V = self.gamma * np.exp(-self.beta * np.sum(
            np.square(_up_diff), axis=2))
        self.upright_V = self.gamma / np.sqrt(2) * np.exp(-self.beta * np.sum(
            np.square(_upright_diff), axis=2))

    def classify_pixels(self):
        self.bgd_indexes = np.where(np.logical_or(
            self.mask == DRAW_BG, self.mask == DRAW_PR_BG))
        self.fgd_indexes = np.where(np.logical_or(
            self.mask == DRAW_FG, self.mask == DRAW_PR_FG))

        assert self.bgd_indexes[0].size > 0
        assert self.fgd_indexes[0].size > 0

        print('(pr_)bgd count: %d, (pr_)fgd count: %d' % (
            self.bgd_indexes[0].size, self.fgd_indexes[0].size))

    def init_GMMs(self):
        self.bgd_gmm = GaussianMixture(self.img[self.bgd_indexes])
        self.fgd_gmm = GaussianMixture(self.img[self.fgd_indexes])

    def assign_GMMs_components(self):
        """Step 1 in Figure 3: Assign GMM components to pixels"""
        self.comp_idxs[self.bgd_indexes] = self.bgd_gmm.which_component(
            self.img[self.bgd_indexes])
        self.comp_idxs[self.fgd_indexes] = self.fgd_gmm.which_component(
            self.img[self.fgd_indexes])

    def learn_GMMs(self):
        """Step 2 in Figure 3: Learn GMM parameters from data z"""
        self.bgd_gmm.fit(self.img[self.bgd_indexes],
                         self.comp_idxs[self.bgd_indexes])
        self.fgd_gmm.fit(self.img[self.fgd_indexes],
                         self.comp_idxs[self.fgd_indexes])

    def construct_gc_graph(self):
        bgd_indexes = np.where(self.mask.reshape(-1) == DRAW_BG)
        fgd_indexes = np.where(self.mask.reshape(-1) == DRAW_FG)
        pr_indexes = np.where(np.logical_or(
            self.mask.reshape(-1) == DRAW_PR_BG, self.mask.reshape(-1) == DRAW_PR_FG))

        print('bgd count: %d, fgd count: %d, uncertain count: %d' % (
            len(bgd_indexes[0]), len(fgd_indexes[0]), len(pr_indexes[0])))

        edges = []
        self.gc_graph_capacity = []

        # t-links
        edges.extend(
            list(zip([self.gc_source] * pr_indexes[0].size, pr_indexes[0])))
        _D = -np.log(self.bgd_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes]))
        self.gc_graph_capacity.extend(_D.tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_sink] * pr_indexes[0].size, pr_indexes[0])))
        _D = -np.log(self.fgd_gmm.calc_prob(self.img.reshape(-1, 3)[pr_indexes]))
        self.gc_graph_capacity.extend(_D.tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_source] * bgd_indexes[0].size, bgd_indexes[0])))
        self.gc_graph_capacity.extend([0] * bgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_sink] * bgd_indexes[0].size, bgd_indexes[0])))
        self.gc_graph_capacity.extend([9 * self.gamma] * bgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_source] * fgd_indexes[0].size, fgd_indexes[0])))
        self.gc_graph_capacity.extend([9 * self.gamma] * fgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_sink] * fgd_indexes[0].size, fgd_indexes[0])))
        self.gc_graph_capacity.extend([0] * fgd_indexes[0].size)
        assert len(edges) == len(self.gc_graph_capacity)

        # print(len(edges))

        # n-links
        img_indexes = np.arange(self.rows * self.cols,
                                dtype=np.uint32).reshape(self.rows, self.cols)

        mask1 = img_indexes[:, 1:].reshape(-1)
        mask2 = img_indexes[:, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.left_V.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        mask1 = img_indexes[1:, 1:].reshape(-1)
        mask2 = img_indexes[:-1, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(
            self.upleft_V.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        mask1 = img_indexes[1:, :].reshape(-1)
        mask2 = img_indexes[:-1, :].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.up_V.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        mask1 = img_indexes[1:, :-1].reshape(-1)
        mask2 = img_indexes[:-1, 1:].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(
            self.upright_V.reshape(-1).tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        assert len(edges) == 4 * self.cols * self.rows - 3 * (self.cols + self.rows) + 2 + \
            2 * self.cols * self.rows

        self.gc_graph = ig.Graph(self.cols * self.rows + 2)
        self.gc_graph.add_edges(edges)

    def estimate_segmentation(self):
        """Step 3 in Figure 3: Estimate segmentation"""
        mincut = self.gc_graph.st_mincut(
            self.gc_source, self.gc_sink, self.gc_graph_capacity)
        print('foreground pixels: %d, background pixels: %d' % (
            len(mincut.partition[0]), len(mincut.partition[1])))
        pr_indexes = np.where(np.logical_or(
            self.mask == DRAW_PR_BG, self.mask == DRAW_PR_FG))
        img_indexes = np.arange(self.rows * self.cols,
                                dtype=np.uint32).reshape(self.rows, self.cols)
        self.mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], mincut.partition[0]),
                                         DRAW_PR_FG, DRAW_PR_BG)
        self.classify_pixels()

    def calc_energy(self):
        U = 0
        for ci in range(self.gmm_components):
            idx = np.where(np.logical_and(self.comp_idxs == ci, np.logical_or(
                self.mask == DRAW_BG, self.mask == DRAW_PR_BG)))
            U += np.sum(-np.log(self.bgd_gmm.coefs[ci] * self.bgd_gmm.calc_score(self.img[idx], ci)))

            idx = np.where(np.logical_and(self.comp_idxs == ci, np.logical_or(
                self.mask == DRAW_FG, self.mask == DRAW_PR_FG)))
            U += np.sum(-np.log(self.fgd_gmm.coefs[ci] * self.fgd_gmm.calc_score(self.img[idx], ci)))

        V = 0
        mask = self.mask.copy()
        mask[np.where(mask == DRAW_PR_BG)] = DRAW_BG
        mask[np.where(mask == DRAW_PR_FG)] = DRAW_FG

        V += np.sum(self.left_V * (mask[:, 1:] == mask[:, :-1]))
        V += np.sum(self.upleft_V * (mask[1:, 1:] == mask[:-1, :-1]))
        V += np.sum(self.up_V * (mask[1:, :] == mask[:-1, :]))
        V += np.sum(self.upright_V * (mask[1:, :-1] == mask[:-1, 1:]))
        return U, V, U + V

    def run(self, num_iters=1, skip_learn_GMMs=False):
        print('skip learn GMMs:', skip_learn_GMMs)
        for _ in range(num_iters):
            if not skip_learn_GMMs:
                self.assign_GMMs_components()
                self.learn_GMMs()
            self.construct_gc_graph()
            self.estimate_segmentation()
            skip_learn_GMMs = False
            # print('data term: %f, smoothness term: %f, total energy: %f' % self.calc_energy())


path_bboxes=r'bboxes'
path_images=r'images'
bboxes_files=os.listdir(path_bboxes)

images=os.listdir(path_images)

assert(len(images)==len(bboxes_files))

bboxes=[]
for bbox_file in bboxes_files:
    with open(os.path.join(path_bboxes,bbox_file)) as file: 
        bboxes.append(file.readline())
        print(file.readline())

rows = 6
num=0
plt.figure(figsize=(20, 20))



#def showGrabCut(img,bbrect):
#    
#    bgdmodel = np.zeros((1,65),np.float64)
#    fgdmodel = np.zeros((1,65),np.float64)
#    img2 = img.copy()                               # a copy of original image
#    mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
#    output = np.zeros(img.shape,np.uint8)     
#    cv2.grabCut(img2,mask,bbrect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
#    mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
#    
#    output = cv2.bitwise_and(img2,img2,mask=mask2)
#    return output
#
#
#images_out=[]
#
#for img_file,bbox in zip(images,bboxes):
#    img=cv2.imread(os.path.join(path_images,img_file))
#    bbox=bbox.split()
#    x1,y1,x2,y2=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
#    rgb = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#    cv2.rectangle(rgb,(x1,y1),(x2,y2),  (0, 255, 0),2)
#    bbrect = (x1,y1,x2,y2)    
#    plt.subplot(rows,5,num+1)
#    plt.title(img_file.split('.')[0])       
#    plt.axis('off')
#    output=showGrabCut(rgb,bbrect)
#    images_out.append(output)
#    plt.imshow(rgb)
#    num=num+1
#    
#rows = 6
#num=0
#plt.figure(figsize=(20, 20))
#    
#for img in images_out:   
#    plt.subplot(rows,5,num+1)
#    plt.axis('off')
#    plt.imshow(img)
#    num=num+1


def showGrabCut(img,bbrect):
    
    img2 = img.copy()                               # a copy of original image
    mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
    output = np.zeros(img.shape,np.uint8)     

    gc = GrabCut(img2, mask, bbrect)
 
    mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
    
    
    output = cv2.bitwise_and(img2,img2,mask=mask2)
    return output



images_gc=[]

for img_file,bbox in zip(images,bboxes):
    img=cv2.imread(os.path.join(path_images,img_file))
    bbox=bbox.split()
    x1,y1,x2,y2=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    rgb = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.rectangle(rgb,(x1,y1),(x2,y2),  (0, 255, 0),2)
    bbrect = (x1,y1,x2,y2)    
    plt.subplot(rows,5,num+1)
    plt.title(img_file.split('.')[0])       
    plt.axis('off')

    output = showGrabCut(rgb, bbrect)
    images_gc.append(output)
    plt.imshow(rgb)
    num=num+1
    
rows = 6
num=0
plt.figure(figsize=(20, 20))
    
for img in images_gc:   
    plt.subplot(rows,5,num+1)
    plt.axis('off')
    plt.imshow(img)
    num=num+1
