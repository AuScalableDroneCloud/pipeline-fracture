# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:28:28 2022

@author: kel321
"""
import cv2
import math
from tqdm import tqdm
import numpy as np
import itertools
from osgeo import gdal
from coshrem.shearletsystem import EdgeSystem, RidgeSystem

import easygui
import matplotlib.pyplot as plt

#reading the image and cheking for georeferencing information
def ReadImage(img_name):
    gray = None
    dataset = None
    georef = False    
    dataset = gdal.Open(img_name, gdal.GA_ReadOnly)
    if dataset:
        if dataset.GetProjection():
            geotransform = dataset.GetGeoTransform()
            georef = True
            if geotransform:
                gray = np.array(dataset.GetRasterBand(1).ReadAsArray())
        else:
            image = cv2.imread(img_name)    
            gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)       
        return( gray)

def SplitInput(text, isInt):
    params = []
    text.replace(" ","")
    p = text.split(",")
    for i in p:
        if isInt:
            params.append(int(i))
        else:
            params.append(float(i))
    return(params)

def ImgSizes(images):
    s = []
    min_s = None
    if len(images) > 0:
        for img in images:
            s.append(img.shape)
        for x,y in itertools.combinations(s, 2):
            if x != y:
                if x < y:
                    min_s = x
                else:
                    min_s = y
        if min_s:
            print('image sizes not consistent')
            print('using', min_s)
        else:
            min_s = s[0]
        return(min_s)
    else:
        return(images[0].shape)
 
def CheckShearletParams(alpha):
    new_alpha = alpha
    for alp in alpha:
        if alp < 1:
           print("invalid entry in alpha: ", alp)
           new_alpha.remove(alp) 
        if alp > 2:
            print("invalid entry in alpha: ", alp)
            new_alpha.remove(alp)          
    return(new_alpha)   
     
def CheckDetectionParams(offset):
    new_offset = offset
    for off in offset:
        if off < 1:
           print("invalid entry in offset: ", off)
           new_offset.remove(off) 
        if off >= 2:
            print("invalid entry in offset: ", off)
            new_offset.remove(off)
    return(new_offset)

#==============================================================================          
def GenerateSystems(p, wavelet_eff_supp, gaussian_eff_supp, scales_per_octave, shear_level, ALPHA, OCTAVES, ridges):
    shearlet_systems = []
    ALPHA = CheckShearletParams(ALPHA)
    all_sys_params = [wavelet_eff_supp, gaussian_eff_supp, scales_per_octave, shear_level, ALPHA, OCTAVES]
    all_sys_combs = list(itertools.product(*all_sys_params)) 
    
    for  param in tqdm(all_sys_combs):
        if ridges:
            print(param[0], param[1], param[2], param[3], param[4], param[5])
            sys = RidgeSystem(*p,
                                wavelet_eff_supp = param[0],
                                gaussian_eff_supp = param[1]
                               # scales_per_octave = param[2],
                               # shear_level = param[3],
                              #  alpha = param[4],
                               # octaves =  param[5]
                                )
            shearlet_systems.append(sys)
        else:
            sys = EdgeSystem(*p,
                              wavelet_eff_supp = param[0],
                              gaussian_eff_supp = param[1],
                              scales_per_octave = param[2],
                              shear_level = param[3],
                              alpha = param[4],
                              octaves =  param[5]
                            )
            shearlet_systems.append(sys)  
    print('generated ', len(shearlet_systems), 'systems')
    return(shearlet_systems)

def DetectFeatures(img_list, shearlet_systems, min_contrast, offset, pivoting_scales, negative, positive, ridges, i_size): 
    feature_img = []
    offset = CheckDetectionParams(offset)
    all_detec_params = [min_contrast, offset]
    all_detec_combs = list(itertools.product(*all_detec_params ))
    print(len(all_detec_combs), " detection combinations.")
    for img in img_list:    
        if img.shape != i_size:
            print('resizing image')
            dim = (i_size[1],i_size[0])
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)    
        detected = np.zeros(img.shape, np.double) 
        for shear_sys in tqdm(shearlet_systems):   
            for detect_param in all_detec_combs:
                if ridges:
                    print(detect_param[0], detect_param[1])
                    features, orientations = shear_sys.detect(img, 
                                                              min_contrast=detect_param[0],  
                                                              offset = detect_param[1], 
                                                              pivoting_scales= pivoting_scales,
                                                              positive_only=positive, 
                                                              negative_only=negative
                                                              )
                else:
                    features, orientations = shear_sys.detect(img, 
                                                              min_contrast=detect_param[0],  
                                                              offset = detect_param[1], 
                                                              pivoting_scales= pivoting_scales
                                                              )
                detected = np.add(detected, features)
        norm = np.zeros(img.shape, np.double)
        normalized = cv2.normalize(detected, norm, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)          
        feature_img.append(detected)
    return(feature_img)
                

filenames  = easygui.fileopenbox("select image file(s)", "CoSh_ensemble", filetypes= "*.jpg", multiple=True)
img_list = []
for f in filenames:
    img_list.append(ReadImage(f))
img_s = ImgSizes(img_list)

wavelet_eff_supp = [5, 10, 50]
gaussian_eff_supp = [2.5, 5, 25]
scales_per_octave = [2, 3]
shear_level = [3]
alpha =  [1, 3]
octaves = [3.5]
ridges = True

min_contrast = [5, 10]
offset = [1]
pivoting_scales = 'all'#all, lowest, highest

systems = GenerateSystems(img_s, wavelet_eff_supp, gaussian_eff_supp, scales_per_octave, shear_level, alpha, octaves, ridges)
F = DetectFeatures(img_list, systems, min_contrast, offset, pivoting_scales, negative = True, positive = False, ridges = True, i_size = img_s)
imgplot = plt.imshow(F[0], cmap=plt.get_cmap('gray'))
plt.show()
