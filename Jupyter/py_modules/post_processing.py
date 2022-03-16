# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:07:01 2022

@author: kel321
"""

import sys
import cv2
import numpy as np
import networkx as nx
from osgeo import gdal, osr, ogr
from skimage.morphology import skeletonize, binary_closing

#==============================================================================
'''
Skeleton Network

build network from nd skeleton image
#https://github.com/Image-Py/sknw

BSD 3-Clause License

Copyright (c) 2017, Yan xiaolong
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

© 2022 GitHub, Inc.
'''
def neighbors(shape):
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])


def mark(img, nbs): # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        if img[p]==0:continue
        s = 0
        for dp in nbs:
            if img[p+dp]!=0:s+=1
        if s==2:img[p]=1
        else:img[p]=2


def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i]//acc[j]
            idx[i] -= rst[i,j]*acc[j]
    rst -= 1
    return rst
    

def fill(img, p, num, nbs, acc, buf):
    img[p] = num
    buf[0] = p
    cur = 0; s = 1; iso = True;
    
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==2:
                img[cp] = num
                buf[s] = cp
                s+=1
            if img[cp]==1: iso=False
        cur += 1
        if cur==s:break
    return iso, idx2rc(buf[:s], acc)

def trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0;
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, idx2rc(buf[:cur+1], acc))
   

def parse_struc(img, nbs, acc, iso, ring):
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in range(len(img)):
        if img[p] == 2:
            isiso, nds = fill(img, p, num, nbs, acc, buf)
            if isiso and not iso: continue
            num += 1
            nodes.append(nds)
    edges = []
    for p in range(len(img)):
        if img[p] <10: continue
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    if not ring: return nodes, edges
    for p in range(len(img)):
        if img[p]!=1: continue
        img[p] = num; num += 1
        nodes.append(idx2rc([p], acc))
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges
    
# use nodes and edges build a networkx graph
def build_graph(nodes, edges, multi=False, full=True):
    os = np.array([i.mean(axis=0) for i in nodes])
    if full: os = os.round().astype(np.uint16)
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=os[i])
    for s,e,pts in edges:
        if full: pts[[0,-1]] = os[[s,e]]
        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()
        graph.add_edge(s,e, pts=pts, weight=l)
    return graph

def mark_node(ske):
    buf = np.pad(ske, (1,1), mode='constant').astype(np.uint16)
    nbs = neighbors(buf.shape)
    mark(buf, nbs)
    return buf
    
def build_sknw(ske, multi=False, iso=True, ring=True, full=True):
    buf = np.pad(ske, (1,1), mode='constant').astype(np.uint16)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    nodes, edges = parse_struc(buf, nbs, acc, iso, ring)
    return build_graph(nodes, edges, multi, full)
    
# draw the graph
def draw_graph(img, graph, cn=255, ce=128):
    acc = np.cumprod((1,)+img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for (s, e) in graph.edges():
        eds = graph[s][e]
        if isinstance(graph, nx.MultiGraph):
            for i in eds:
                pts = eds[i]['pts']
                img[np.dot(pts, acc)] = ce
        else: img[np.dot(eds['pts'], acc)] = ce
    for idx in graph.nodes():
        pts = graph.nodes[idx]['pts']
        img[np.dot(pts, acc)] = cn
#==============================================================================

def SigmoidNonlinearity(image):
    ridges_norm_sig = np.zeros(image.shape, np.double)
    w,h = image.shape
    for i in range(w):
        for j in range(h):
            if image[i][j] != 0:
                ridges_norm_sig[i][j] = 1 / (1 + np.exp((-1)*image[i][j]))
    return(ridges_norm_sig)

def Threshholding(image, thresh, ksize, alpha, beta):   
    w,h = image.shape
    for i in range(w):
        for j in range(h):
            if image[i][j] < thresh:
                image[i][j] = 0
    thresh_sig_img = SigmoidNonlinearity(image) 
    int_img = (np.multiply(thresh_sig_img, 255)).astype(np.uint8)   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
    opening = cv2.morphologyEx(int_img, cv2.MORPH_OPEN, kernel)
    clean =  int_img - opening
    adjusted = cv2.convertScaleAbs(clean, alpha=alpha, beta=beta)
    return(adjusted)

def CleanUp(image, connectivity, min_size):
    img = np.array(image).astype(np.uint8)
    
    #img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape))
    for i in range(nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
    skeleton = binary_closing(skeletonize(img2))
    return(skeleton)
    
def WritePoints2SHP(graph, outputFile, refWKT = None, tolerance=1e-3): 
    #first convert the graph vertices into numpy array
    nodes = graph.nodes()
    points = np.array([nodes[i]['o'] for i in nodes])
    degrees = []
    for i in range(len(nodes)):
        degrees.append(graph.degree[i])
    #get the GDAL driver for ESRI shapefile
    driverName = "ESRI Shapefile"
    drv = gdal.GetDriverByName( driverName )
    if drv is None:
        print ("%s driver not available.\n" % driverName)
        sys.exit( 1 )
        
#  Create the shp-file (or open it if it exists)
    ds = drv.Create(outputFile, 0, 0, 0, gdal.GDT_Unknown )
    if ds is None:
        print ("Creation of output file failed. Trying to open file...\n")
        ds = drv.Open(outputFile, 1) # 0 means read-only. 1 means writeable.
        if ds is None:
            print ("Creation of output file failed.\n")
            sys.exit( 1 )
# get layer and check if reference is given (write non-georeferenced shp if not defined)     
    lyr = ds.GetLayer(0)
    if lyr is None:
        if refWKT:
            print("refWKT")
            SpatialRef = osr.SpatialReference()
            SpatialRef.ImportFromWkt(refWKT)
            lyr = ds.CreateLayer( "graph_vertices", SpatialRef, ogr.wkbPoint)  
        else:
            lyr = ds.CreateLayer( "graph_vertices", None, ogr.wkbPoint)
        if lyr is None:
            print ("Layer creation failed.\n")
            sys.exit( 1 )    
    lyr.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    lyr.CreateField(ogr.FieldDefn('degree', ogr.OFTInteger))
#loop though the ponts and create lines in the layer
    for i, p in enumerate(points): 
        if degrees[i] > 0:
            feat = ogr.Feature( lyr.GetLayerDefn())
            feat.SetField( "id", i )
            feat.SetField( "degree", degrees[i] )
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(float(p[0]), float(p[1]))
            #point.SwapXY
            feat.SetGeometry(point)
            lyr.CreateFeature(feat) 
            feat.Destroy()
    ds = None

def WritePolyline2SHP(graph, outputFile, refWKT = None, tolerance=1e-3): 
    #first convert the graph edges into a list of numpy arrays lists
    poly_lines = []
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']      
        poly_lines.append([ps[:,0], ps[:,1]])
           
#get the GDAL driver for ESRI shapefile
    driverName = "ESRI Shapefile"
    drv = gdal.GetDriverByName( driverName )
    if drv is None:
        print ("%s driver not available.\n" % driverName)
        sys.exit( 1 )
        
#  Create the shp-file (or open it if it exists)
    ds = drv.Create(outputFile, 0, 0, 0, gdal.GDT_Unknown )
    if ds is None:
        print ("Creation of output file failed. Trying to open file...\n")
        ds = drv.Open(outputFile, 1) # 0 means read-only. 1 means writeable.
        if ds is None:
            print ("Creation of output file failed.\n")
            sys.exit( 1 )
            
# get layer and check if reference is given (write non-georeferenced shp if not defined)     
    lyr = ds.GetLayer(0)
    if lyr is None:
        if refWKT:
            print("refWKT")
            SpatialRef = osr.SpatialReference()
            SpatialRef.ImportFromWkt(refWKT)
            lyr = ds.CreateLayer( "fitted_polylines", SpatialRef, ogr.wkbLineString)  
        else:
            lyr = ds.CreateLayer( "fitted_polylines", None, ogr.wkbLineString)
        if lyr is None:
            print ("Layer creation failed.\n")
            sys.exit( 1 )    
    lyr.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))      

#loop though the polylines and create lines in the layer
    for i, l in enumerate(poly_lines): 
        feat = ogr.Feature( lyr.GetLayerDefn())
        feat.SetField( "id", i )
        line = ogr.Geometry(ogr.wkbLineString)
        for n, p in enumerate(l):
            a_x = np.array(l[0])
            a_y = np.array(l[1])
            point_list = []
        for nn, pp in enumerate(a_x):
            point_list.append([float(a_x[nn]), float(a_y[nn])])
        np.unique(point_list)
        for point in point_list:
            line.AddPoint(point[0], point[1])  #need to flip x coodinates in case of non-georeferenced data
        simpleLine = line.Simplify(tolerance)   #simplyfy the line geometry
        #simpleLine.SwapXY
        feat.SetGeometry(simpleLine)
        lyr.CreateFeature(feat) 
        feat.Destroy()
    ds = None