# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:28:28 2022

@author: kel321
"""
import cv2
import time
import numpy as np
import itertools
from osgeo import gdal
from coshrem.shearletsystem import EdgeSystem, RidgeSystem
from concurrent.futures import ProcessPoolExecutor

#reading the image and cheking for georeferencing information
def ReadImage(img_name):
    dataset = None
    dataset = gdal.Open(img_name, gdal.GA_ReadOnly)
    if dataset:
        if dataset.GetProjection():
            if dataset.GetGeoTransform():
                gray = np.array(dataset.GetRasterBand(1).ReadAsArray())
                return((gray, dataset))
        else:
            gray = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)         
            if gray.any() != None:
                if len(gray.shape)==3:
                    gray  = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)      
            return((gray, None))

#write the images 
def WriteImage(img_list, features, filename):
    assert len(img_list) == len(features),"number of images not equal to number of ridge/edge ensembles"
    for i, img in enumerate(img_list):
        outfile = filename + "_" + str(i) + ".tif"
        if img[1] != None:
            dataset = img[1]          
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(outfile, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
            outdata.SetGeoTransform(dataset.GetGeoTransform())
            outdata.SetProjection(dataset.GetProjection())
            outdata.GetRasterBand(1).WriteArray(features[i])
            outdata.FlushCache() 
        else:
            cv2.imwrite(outfile, features[i])
 
#split input from text boxes
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

#check consient size of images adn retun dimesnion of samllest image
def ImgSizes(images):
    s = []
    min_s = None
    if len(images) > 0:
        for img in images:
            s.append(img[0].shape)
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
 
#check for invalid parameters for shearlet system generation
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
 
#Check for invalid detection parameters    
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

def GetRidgeSys(param):
    pp = list(param)    
   # print(pp[0][1], pp[0][2], pp[0][3], pp[0][4], pp[0][5], pp[0][6])
    sys = RidgeSystem(*pp[0][0], wavelet_eff_supp = pp[0][1], 
                      gaussian_eff_supp = pp[0][2], 
                      scales_per_octave = pp[0][3],
                      shear_level = pp[0][4],
                      alpha = pp[0][5],
                      octaves =  pp[0][6]
                      ) 
    return(sys)

def GetEdgeSys(param):
    pp = list(param)    
    #print(pp[0][1], pp[0][2], pp[0][3], pp[0][4], pp[0][5], pp[0][6])
    sys = EdgeSystem(*pp[0][0], wavelet_eff_supp = pp[0][1], 
                      gaussian_eff_supp = pp[0][2], 
                      scales_per_octave = pp[0][3],
                      shear_level = pp[0][4],
                      alpha = pp[0][5],
                      octaves =  pp[0][6]
                      ) 
    return(sys)

#Generate Shearlet systems          
def GenerateSystems(i_size, wavelet_eff_supp, gaussian_eff_supp, scales_per_octave, shear_level, ALPHA, OCTAVES, ridges): 
    t = time.time()
    params = []
    shearlet_systems = []
    all_sys_params = [wavelet_eff_supp, gaussian_eff_supp, scales_per_octave, shear_level, ALPHA, OCTAVES]
    all_sys_combs = list(itertools.product(*all_sys_params)) 
    print("generating ", len(all_sys_combs), " systems.")
    for  param in all_sys_combs:
        params.append([i_size, param[0], param[1], param[2],param[3],param[4],param[5]])
    iter_param = zip(params)
    with ProcessPoolExecutor(max_workers=8) as executor:
        if ridges:
            for r in executor.map(GetRidgeSys, iter_param):
                shearlet_systems.append(r) 
        else:
            for r in executor.map(GetRidgeSys, iter_param):
                shearlet_systems.append(r) 
    elapsed = time.time() - t
    print(" done in ", elapsed, "s")   
    return(shearlet_systems)

def Detect(params):  
    pp = list(params)
    sys = pp[0][0]
    img = pp[0][1]
    min_contrast = pp[0][2]
    offset = pp[0][3]
    pivoting = pp[0][4]
    negative = pp[0][5]
    positive = pp[0][6]
    ridges = pp[0][7]
    #print(parameters[0][0], parameters[0][1])
    if ridges:
        features, orientations = sys.detect(img, 
                                            min_contrast = min_contrast,  
                                            offset = offset, 
                                            pivoting_scales = pivoting,
                                            positive_only = positive, 
                                            negative_only = negative
                                            )
    else:
        features, orientations = sys.detect(img, 
                                            min_contrast = min_contrast,  
                                            offset = offset, 
                                            pivoting_scales = pivoting,
                                            )     
    Msum = np.sum(features)
    has_nan = np.isnan(Msum)
    if (has_nan):
        print("NaN in array, check parameter combination.")
        f_ret = np.zeros(img[0].shape, np.double) 
    else:        
        f_ret = features
    return(f_ret)




#Detect features in image
def DetectFeatures(img_list, shearlet_systems, min_contrast, offset, pivoting_scales, negative, positive, ridges, i_size): 
    print('detecting features with ', len(shearlet_systems), " systems.")
    t = time.time()
    feature_img = []
    # offset = CheckDetectionParams(offset)
    all_detec_params = [min_contrast, offset]
    all_detec_combs = list(itertools.product(*all_detec_params ))
    print(len(all_detec_combs), " detection combinations.")
    for i, img in enumerate(img_list):    
         detected = np.zeros(img[0].shape, np.double)   
         
         func_params = []
         
         for detect in all_detec_combs:
             for shear_sys in shearlet_systems:   
                 func_params.append((shear_sys, img[0], detect[0], detect[1], pivoting_scales, negative, positive, ridges))
                 
         fp = zip(func_params)
         with ProcessPoolExecutor(max_workers=8) as executor:
             for r in executor.map(Detect, fp): 
                 detected = np.add(detected, r)
         norm = np.zeros(img[0].shape, np.double)
         normalized = cv2.normalize(detected, norm, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)          
         feature_img.append(normalized)
    elapsed = time.time() - t
    print(" done in ", elapsed, "s")  
    return(feature_img)