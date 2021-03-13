# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:08:53 2019

@author: E442282
"""

import numpy as np
import scipy
from scipy.misc import imread
import maxflow
import matplotlib.pyplot as plt
import cv2

#img = cv2.imread(r"a.png")
#
#plt.imshow(img,cmap='gray')

img = cv2.imread(r"a2.png")

plt.imshow(img,cmap='gray')


# Create the graph.
g = maxflow.Graph[int]()

#Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(img.shape)

# Add non-terminal edges with the same capacity.
g.add_grid_edges(nodeids, 50)
# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
g.add_grid_tedges(nodeids, img, 255-img)

# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
## Show the result.
#
plt.imshow(img2,cmap='gray')
#plt.show()