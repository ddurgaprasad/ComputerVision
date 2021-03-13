from PIL import Image
import numpy as np
import random

from math import log
import cv2
import copy
import matplotlib.pyplot as plt

import ctypes

#https://github.com/guyuchao/Grabcut

class Kmeans(object):
    def __init__(self,images,dim=3,cluster=5,epoches=2):
        self.images=images.reshape(-1,dim)
        self.pixelnum=self.images.shape[0]
        self.cluster=cluster
        self.belong=np.zeros(self.pixelnum)
        self.cluster_centers=self.images[random.sample(range(self.pixelnum),self.cluster)]
        self.epoches=epoches

    def run(self):
        for i in range(self.epoches):
            self.updates_belonging()
            self.updates_centers()
        return self.belong

    def updates_belonging(self):
        newbelong=np.zeros(self.pixelnum)
        for num in range(self.pixelnum):
            cost=[np.square(self.images[num]-self.cluster_centers[i]).sum() for i in range(self.cluster)]
            newbelong[num]=np.argmin(cost)
        self.belong=newbelong

    def updates_centers(self):
        num_clusters=np.zeros(self.cluster)
        for cluster_idx in range(self.cluster):
            belong_to_cluster=np.where(self.belong==cluster_idx)[0]

            num_cluster=len(belong_to_cluster)
            num_clusters[cluster_idx]=num_cluster

        for cluster_idx in range(self.cluster):
            if num_clusters[cluster_idx]==0:
                #find max
                max_cluster=np.argmax(num_clusters)
                belong_to_cluster=np.where(self.belong==max_cluster)[0]
                pixels=self.images[belong_to_cluster]
                cost=[np.square(self.images[id]-self.cluster_centers[max_cluster]).sum() for id in range(self.pixelnum)]
                far_pixel_idx=np.argmax(cost)
                self.belong[far_pixel_idx]=cluster_idx
                self.cluster_centers[cluster_idx]=self.images[cluster_idx]
            else:
                idx=np.where(self.belong==cluster_idx)[0]
                self.cluster_centers[cluster_idx]=self.images[idx].sum(0)/len(idx)
    def plot(self):
        data_x = []
        data_y = []
        data_z = []
        for i in range(self.cluster):
            index = np.where(self.belong == i)
            data_x.extend(self.images[index][:, 0].tolist())
            data_y.extend(self.images[index][:, 1].tolist())
            data_z.extend([i / self.cluster for j in range(len(list(index)[0]))])
        sc = plt.scatter(data_x, data_y, c=data_z, vmin=0, vmax=1, s=35, alpha=0.8)
        plt.colorbar(sc)
        plt.show()


'''
if __name__=='__main__':
    A=np.random.random((1000,20,2))
    kmeans=Kmeans(A,2,20,10)
    kmeans()
    kmeans.plot()
'''

class Pointer:
    def __init__(self, var):
        self.id = id(var)

    def get_value(self):
        return ctypes.cast(self.id, ctypes.py_object).value


class Vertex:
    def __init__(self):
        self.next = 0 # Initialized and used in maxflow() only
        self.parent = 0
        self.first = 0
        self.ts = 0
        self.dist = 0
        self.weight = 0
        self.t = 0

class Edge:
    def __init__(self):
        self.dst = 0
        self.next = 0
        self.weight = 0.0

class GCGraph:
    def __init__(self, vertex_count, edge_count):
        self.vertexs = []
        self.edges = []
        self.flow = 0
        self.vertex_count = vertex_count
        self.edge_count = edge_count

    def add_vertex(self):
        v = Vertex()
        self.vertexs.append(v)
        return len(self.vertexs) -  1

    def add_edges(self, i, j, w, revw):

        a = len(self.edges)
        # As is said in the C++ code, if edges.size() == 0, then resize edges to 2.

        # if a == 0:
        # 	a = 2

        fromI = Edge()
        fromI.dst = j
        fromI.next = self.vertexs[i].first
        fromI.weight = w
        self.vertexs[i].first = a
        self.edges.append(fromI)

        toI = Edge()
        toI.dst = i
        toI.next = self.vertexs[j].first
        toI.weight = revw
        self.vertexs[j].first = a + 1
        self.edges.append(toI)



    def add_term_weights(self, i, source_weight, sink_weight):
        dw = self.vertexs[i].weight
        if dw > 0:
            source_weight += dw
        else:
            sink_weight -= dw
        self.flow += source_weight if source_weight < sink_weight else sink_weight
        self.vertexs[i].weight = source_weight - sink_weight

    def max_flow(self):
        TERMINAL = -1
        ORPHAN = -2
        stub = Vertex()
        nilNode = Pointer(stub)
        first = Pointer(stub)
        last = Pointer(stub)
        curr_ts = 0
        stub.next = nilNode.get_value()
        # # print(first.get_value() == nilNode.get_value())

        orphans = []

        # initialize the active queue and the graph vertices
        for i in range(len(self.vertexs)):
            v = self.vertexs[i]
            v.ts = 0
            if v.weight != 0:
                last.get_value().next = v
                last.id = id(v)
                v.dist = 1
                v.parent = TERMINAL
                v.t = v.weight < 0
            else:
                v.parent = 0
            # # print(first.get_value().next == nilNode.get_value())
        first.id = id(first.get_value().next)
        last.get_value().next = nilNode.get_value()
        nilNode.get_value().next = 0
        # # print(first.get_value() == nilNode.get_value())

        # count = 0
        # Search Path -> Augment Graph -> Restore Trees
        while True:
            # print('1','\n', [x.t for x in self.vertexs])

            # count += 1
            # # print(count)
            e0 = -1
            ei = 0
            ej = 0

            while first.get_value() != nilNode.get_value():
                v = first.get_value()
                if v.parent:
                    vt = v.t
                    ei = v.first
                    while ei != 0:
                        if self.edges[ei^vt].weight == 0:
                            ei = self.edges[ei].next
                            continue
                        u = self.vertexs[self.edges[ei].dst]
                        if not u.parent:
                            u.t = vt
                            u.parent = ei ^ 1
                            u.ts = v.ts
                            u.dist = v.dist + 1
                            if not u.next:
                                u.next = nilNode.get_value()
                                last.get_value().next = u
                                last.id = id(u)
                            ei = self.edges[ei].next
                            continue
                        if u.t != vt:
                            e0 = ei ^ vt
                            break
                        if u.dist > v.dist + 1 and u.ts <= v.ts:
                            u.parent = ei ^ 1
                            u.ts = v.ts
                            u.dist = v.dist + 1
                        # # print(self.edges[ei].next)
                        ei = self.edges[ei].next
                    if e0 > 0:
                        break
                first.id = id(first.get_value().next)
                # first = first.next
                v.next = 0

            # print('2','\n', [x.t for x in self.vertexs])

            if e0 <= 0:
                break

            minWeight = self.edges[e0].weight
            for k in range(1, -1, -1):
                v = self.vertexs[self.edges[e0^k].dst]
                while True:
                    # # print('f')
                    ei = v.parent
                    if ei < 0:
                        break
                    weight = self.edges[ei^k].weight
                    minWeight = min(minWeight, weight)
                    v = self.vertexs[self.edges[ei].dst]
                weight = abs(v.weight)
                minWeight = min(minWeight, weight)

            self.edges[e0].weight -= minWeight
            self.edges[e0^1].weight += minWeight
            self.flow += minWeight

            for k in range(1, -1, -1):
                v = self.vertexs[self.edges[e0^k].dst]
                while True:
                    # # print('d')
                    ei = v.parent
                    if ei < 0:
                        break
                    self.edges[ei^(k^1)].weight += minWeight
                    self.edges[ei^k].weight -= minWeight
                    if self.edges[ei^k].weight == 0:
                        orphans.append(v)
                        v.parent = ORPHAN
                    v = self.vertexs[self.edges[ei].dst]
                v.weight = v.weight + minWeight*(1-k*2)
                if v.weight == 0:
                    orphans.append(v)
                    v.parent = ORPHAN
            curr_ts += 1

            while len(orphans) != 0:
                # v2 = orphans[-1]
                # print('v', v2)
                v2 = orphans.pop()
                minDist = float('inf')
                e0 = 0
                vt = v2.t

                ei = v2.first
                bcount = 0
                while ei != 0:
                    bcount += 1
                    # print('1', bcount)
                    # print(self.edges[ei^(vt^1)].weight)
                    if self.edges[ei^(vt^1)].weight == 0:
                        ei = self.edges[ei].next
                        continue
                    u = self.vertexs[self.edges[ei].dst]
                    if u.t != vt or u.parent == 0:
                        ei = self.edges[ei].next
                        continue

                    d = 0
                    while True:
                        # bcount += 1
                        # print(bcount)
                        if u.ts == curr_ts:
                            d += u.dist
                            break
                        ej = u.parent
                        d += 1
                        # print(d)
                        if ej < 0:
                            if ej == ORPHAN:
                                d = float('inf') - 1
                            else:
                                u.ts = curr_ts
                                u.dist = 1
                            break
                        u = self.vertexs[self.edges[ej].dst]
                    # print(ei)
                        # print('u', u)
                    # # print('aaa')

                    d += 1
                    # print(d == float('inf'))
                    if d < float("inf"):
                        if d < minDist:
                            minDist = d
                            e0 = ei
                        u = self.vertexs[self.edges[ei].dst]
                        while u.ts != curr_ts:
                            # print(u.ts)
                            u.ts = curr_ts
                            d -= 1
                            u.dist = d
                            u = self.vertexs[self.edges[u.parent].dst]

                    ei = self.edges[ei].next
                    # print(ei)

                # print('aaabb')
                v2.parent = e0
                if v2.parent > 0:
                    v2.ts = curr_ts
                    v2.dist = minDist
                    continue

                v2.ts = 0
                ei = v2.first
                while ei != 0:
                    # print('a')
                    u = self.vertexs[self.edges[ei].dst]
                    ej = u.parent
                    if u.t != vt or (not ej):
                        ei = self.edges[ei].next
                        continue
                    if self.edges[ei^(vt^1)].weight and (not u.next):
                        u.next = nilNode.get_value()
                        # last = last.next = u
                        last.get_value().next = u
                        last.id = id(u)
                    if ej > 0 and self.vertexs[self.edges[ej].dst] == v2:
                        orphans.append(u)
                        u.parent = ORPHAN
                    ei = self.edges[ei].next
                # print(orphans)
        # # print([self.vertexs[i].t for i in range(len(self.vertexs))])

        return self.flow

    def insource_segment(self, i):
        return self.vertexs[i].t == 0


class GMM(object):
    def __init__(self,cluster=5):
        self.cluster=cluster
        self.weight=np.zeros(cluster)#pai x in the paper
        self.means=np.zeros((cluster,3))
        self.covs=np.zeros((cluster,3,3))
        self.inverse_cov = np.zeros((cluster, 3, 3))
        self.delta_cov=np.zeros(cluster)

        self.sums_for_mean=np.zeros((cluster,3))
        self.product_for_cov=np.zeros((cluster,3,3))
        self.pixel_counts=np.zeros(cluster)
        self.pixel_total_count=0

    def init(self):
        self.sums_for_mean = np.zeros((self.cluster, 3))
        self.product_for_cov = np.zeros((self.cluster, 3, 3))
        self.pixel_counts = np.zeros(self.cluster)
        self.pixel_total_count = 0

    def add_pixel(self, pixel, cluster):
        pixel_c=copy.deepcopy(pixel)
        cluster=int(cluster)
        self.sums_for_mean[cluster] += pixel_c
        pixel_c=pixel_c[np.newaxis,:]
        self.product_for_cov[cluster] += np.dot(np.transpose(pixel_c),pixel_c)
        self.pixel_counts[cluster] += 1
        self.pixel_total_count += 1

    def learning(self):
        variance = 0.01
        for cluster in range(self.cluster):
            n=self.pixel_counts[cluster]
            if n!=0:
                self.weight[cluster]=n/self.pixel_total_count
                self.means[cluster]=self.sums_for_mean[cluster]/n
                tmp_mean=copy.deepcopy(self.means[cluster])
                tmp_mean=tmp_mean[np.newaxis,:]
                productmean=np.dot(np.transpose(tmp_mean),tmp_mean)
                self.covs[cluster]=self.product_for_cov[cluster]/n-productmean
                self.delta_cov[cluster]=np.linalg.det(self.covs[cluster])
            while self.delta_cov[cluster]<=0:
                self.covs[cluster]+=np.eye(3,3)*variance
                self.delta_cov[cluster]=np.linalg.det(self.covs[cluster])
            self.inverse_cov[cluster]=np.linalg.inv(self.covs[cluster])
                #print(np.dot(self.covs[cluster],self.inverse_cov[cluster]))
        self.init()
    def pred_cluster(self,cluster,pixel):

        if self.weight[cluster]>0:
            diff=copy.deepcopy(pixel)-self.means[cluster]
            mult=((np.transpose(self.inverse_cov[cluster])*diff).sum(1)*diff).sum()
            res = 1.0 / np.sqrt(self.delta_cov[cluster]) * np.exp(-0.5* mult)
            return res
        else:
            return 0

    def pixel_from_cluster(self,pixel):
        p=np.array([self.pred_cluster(cluster,pixel) for cluster in range(self.cluster)])
        return p.argmax()

    def pred_GMM(self,pixel):
        res=np.array([self.weight[cluster]*self.pred_cluster(cluster,pixel) for cluster in range(self.cluster)])
        assert res.sum()>0,"error"
        return res.sum()
from sklearn.mixture import GaussianMixture


class grabcut(object):
    def __init__(self):
        self.cluster=5
        self.BGD_GMM=None
        self.FGD_GMM=None
        self._gamma=50
        self._lambda=9*self._gamma
        self.GT_bgd=0#ground truth background
        self.P_fgd=1#ground truth foreground
        self.P_bgd=2#may be background
        self.GT_fgd=3#may be foreground

    def calcBeta(self,npimg):
        '''

        :param self:
        :param npimg:array of img:h,w,c,type=np.float32
        :return: beta :reference to formula 5 of 《https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf》
        '''
        rows,cols=npimg.shape[:2]

        ldiff = np.linalg.norm(npimg[:, 1:] - npimg[:, :-1])
        uldiff = np.linalg.norm(npimg[1:, 1:] - npimg[:-1, :-1])
        udiff = np.linalg.norm(npimg[1:, :] - npimg[:-1, :])
        urdiff = np.linalg.norm(npimg[1:, :-1] - npimg[:-1, 1:])
        beta=np.square(ldiff)+np.square(uldiff)+np.square(udiff)+np.square(urdiff)
        beta = 1 / (2 * beta / (4 * cols * rows - 3 * cols - 3 * rows + 2))
        return beta

    def calcSmoothness(self, npimg, beta, gamma):
        rows,cols=npimg.shape[:2]
        self.lweight = np.zeros([rows, cols])
        self.ulweight = np.zeros([rows, cols])
        self.uweight = np.zeros([rows, cols])
        self.urweight = np.zeros([rows, cols])
        for y in range(rows):
            for x in range(cols):
                color = npimg[y, x]
                if x >= 1:
                    diff = color - npimg[y, x-1]
                    # print(np.exp(-self.beta*(diff*diff).sum()))
                    self.lweight[y, x] = gamma*np.exp(-beta*(diff*diff).sum())
                if x >= 1 and y >= 1:
                    diff = color - npimg[y-1, x-1]
                    self.ulweight[y, x] = gamma/np.sqrt(2) * np.exp(-beta*(diff*diff).sum())
                if y >= 1:
                    diff = color - npimg[y-1, x]
                    self.uweight[y, x] = gamma*np.exp(-beta*(diff*diff).sum())
                if x+1 < cols and y >= 1:
                    diff = color - npimg[y-1, x+1]
                    self.urweight[y, x] = gamma/np.sqrt(2)*np.exp(-beta*(diff*diff).sum())

    def init_with_kmeans(self,npimg,mask):
        self._beta = self.calcBeta(npimg)
        self.calcSmoothness(npimg, self._beta, self._gamma)

        bgd = np.where(mask==self.GT_bgd)
        prob_fgd = np.where(mask==self.P_fgd)
        BGDpixels = npimg[bgd]#(_,3)
        FGDpixels = npimg[prob_fgd]#(_,3)

        KmeansBgd = Kmeans(BGDpixels, dim=3, cluster=5, epoches=2)
        
        KmeansFgd = Kmeans(FGDpixels, dim=3, cluster=5, epoches=2)

        bgdlabel=KmeansBgd.run() # (BGDpixel.shape[0],1)
        fgdlabel=KmeansFgd.run() # (FGDpixel.shape[0],1)

        self.BGD_GMM = GMM()  # The GMM Model for BGD
        self.FGD_GMM = GMM()  # The GMM Model for FGD


        for idx,label in enumerate(bgdlabel):
            self.BGD_GMM.add_pixel(BGDpixels[idx],label)
        for idx, label in enumerate(fgdlabel):
            self.FGD_GMM.add_pixel(FGDpixels[idx], label)

        self.BGD_GMM.learning()
        self.FGD_GMM.learning()

    def __call__(self,epoches,npimg,mask):
        self.init_with_kmeans(npimg,mask)
        for epoch in range(epoches):
            self.assign_step(npimg,mask)
            self.learn_step(npimg,mask)
            self.construct_gcgraph(npimg,mask)
            mask = self.estimate_segmentation(mask)
            img = copy.deepcopy(npimg)
            img[np.logical_or(mask == self.P_bgd, mask == self.GT_bgd)] = 0
        return Image.fromarray(img.astype(np.uint8))

    def assign_step(self,npimg,mask):
        rows,cols=npimg.shape[:2]
        clusterid=np.zeros((rows,cols))
        for row in range(rows):
            for col in range(cols):
                pixel=npimg[row,col]
                if mask[row,col]==self.GT_bgd or mask[row,col]==self.P_bgd:#bgd
                    clusterid[row,col]=self.BGD_GMM.pixel_from_cluster(pixel)
                else:
                    clusterid[row, col] = self.FGD_GMM.pixel_from_cluster(pixel)
        self.clusterid=clusterid.astype(np.int)

    def learn_step(self,npimg,mask):
        for cluster in range(self.cluster):
            bgd_cluster=np.where(np.logical_and(self.clusterid==cluster,np.logical_or(mask==self.GT_bgd,mask==self.P_bgd)))
            fgd_cluster=np.where(np.logical_and(self.clusterid==cluster,np.logical_or(mask==self.GT_fgd,mask==self.P_fgd)))
            for pixel in npimg[bgd_cluster]:
                self.BGD_GMM.add_pixel(pixel,cluster)
            for pixel in npimg[fgd_cluster]:
                self.FGD_GMM.add_pixel(pixel,cluster)
        self.BGD_GMM.learning()
        self.FGD_GMM.learning()


    def construct_gcgraph(self,npimg,mask):
        rows,cols=npimg.shape[:2]
        vertex_count = rows*cols
        edge_count = 2 * (4 * vertex_count - 3 * (rows + cols) + 2)
        self.graph = GCGraph(vertex_count, edge_count)
        for row in range(rows):
            for col in range(cols):
                #source background sink foreground
                vertex_index = self.graph.add_vertex()
                color = npimg[row, col]
                if mask[row, col] == self.P_bgd or mask[row, col] == self.P_fgd:#pred fgd
                    fromSource = -log(self.BGD_GMM.pred_GMM(color))
                    toSink = -log(self.FGD_GMM.pred_GMM(color))
                elif mask[row, col] == self.GT_bgd:
                    fromSource = 0
                    toSink = self._lambda
                else:
                    fromSource = self._lambda
                    toSink = 0
                self.graph.add_term_weights(vertex_index, fromSource, toSink)

                if col-1 >= 0:
                    w = self.lweight[row, col]
                    self.graph.add_edges(vertex_index, vertex_index - 1, w, w)
                if row-1 >= 0 and col-1 >= 0:
                    w = self.ulweight[row, col]
                    self.graph.add_edges(vertex_index, vertex_index - cols - 1, w, w)
                if row-1 >= 0:
                    w = self.uweight[row, col]
                    self.graph.add_edges(vertex_index, vertex_index - cols, w, w)
                if col+1 < cols and row-1 >= 0:
                    w = self.urweight[row, col]
                    self.graph.add_edges(vertex_index, vertex_index - cols + 1, w, w)

    def estimate_segmentation(self,mask):
        rows,cols=mask.shape
        self.graph.max_flow()
        for row in range(rows):
            for col in range(cols):
                if mask[row, col] == self.P_fgd or mask[row,col]==self.P_bgd :
                    if self.graph.insource_segment(row * cols + col):  # Vertex Index
                        mask[row, col] = self.P_fgd
                    else:
                        mask[row, col] = self.P_bgd

        return mask

import os
path_bboxes=r'bboxes'
path_images=r'images'
bboxes_files=os.listdir(path_bboxes)

images=os.listdir(path_images)

assert(len(images)==len(bboxes_files))
bboxes=[]
for bbox_file in bboxes_files:
    with open(os.path.join(path_bboxes,bbox_file)) as file: 
        bboxes.append(file.readline())
        
images_out=[]

for img_file,bbox in zip(images,bboxes):
    img=cv2.imread(os.path.join(path_images,img_file))
    bbox=bbox.split()
    x1,y1,x2,y2=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    rgb = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    images_out.append(rgb)

if __name__=="__main__":
    filename=r"testlena.jpg"
    img= cv2.imread(filename)
    gg=grabcut()
    mask = np.zeros(img.shape[:2])
#    left = 34
#    right = 315
#    top = 66
#    bottom = 374
#    mask[left:right, top:bottom] = gg.P_fgd
#    myimg=gg(epoches=1,npimg=img,mask=mask)
##    cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
##    plt.imshow(img)
#    plt.imshow(myimg)
    
    
    image_no=bboxes_files.index('person1.txt')
    
    mask = np.zeros(images_out[image_no].shape[:2])
    bboxes[image_no].split()
    x1,y1,x2,y2=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    mask[x1:x2, y1:y2] = gg.P_fgd
    myimg=gg(epoches=1,npimg=images_out[image_no],mask=mask)
    plt.imshow(myimg)
    
    
    
    
from matplotlib import pyplot as plt
from sklearn import mixture
filename=r'C:\SAI\IIIT\2019_Spring\Assignment4\images\sheep.jpg'
X = cv2.imread(filename)
old_shape = X.shape
X = X.reshape(-1,3)
gmm = mixture.GaussianMixture(covariance_type='full', n_components=2)
gmm.fit(X)
clusters = gmm.predict(X)
clusters = clusters.reshape(old_shape[0], old_shape[1])
plt.imshow(clusters)    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    