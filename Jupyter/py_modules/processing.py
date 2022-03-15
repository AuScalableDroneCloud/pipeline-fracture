# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:28:28 2022

@author: kel321
"""
import cv2
from tqdm import tqdm
import numpy as np
import itertools
from osgeo import gdal
from coshrem.shearletsystem import EdgeSystem, RidgeSystem

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
            image = cv2.imread(img_name)    
            gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)       
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

#checj consient size of images adn retun dimesnion of samllest image
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

#Generate Shearlet systems          
def GenerateSystems(p, wavelet_eff_supp, gaussian_eff_supp, scales_per_octave, shear_level, ALPHA, OCTAVES, ridges):
    shearlet_systems = []
    ALPHA = CheckShearletParams(ALPHA)
    all_sys_params = [wavelet_eff_supp, gaussian_eff_supp, scales_per_octave, shear_level, ALPHA, OCTAVES]
    all_sys_combs = list(itertools.product(*all_sys_params)) 
    
    for  param in tqdm(all_sys_combs):
        if ridges:
           # print(param[0], param[1], param[2], param[3], param[4], param[5])
            sys = RidgeSystem(*p,
                                wavelet_eff_supp = param[0],
                                gaussian_eff_supp = param[1],
                                scales_per_octave = param[2],
                                shear_level = param[3],
                                alpha = param[4],
                                octaves =  param[5]
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

#Detect features in image
def DetectFeatures(img_list, shearlet_systems, min_contrast, offset, pivoting_scales, negative, positive, ridges, i_size): 
    feature_img = []
    offset = CheckDetectionParams(offset)
    all_detec_params = [min_contrast, offset]
    all_detec_combs = list(itertools.product(*all_detec_params ))
    print(len(all_detec_combs), " detection combinations.")
    for img in img_list:    
        resized = False
        detected = np.zeros(img[0].shape, np.double) 
    
        if img[0].shape != i_size:
            old_dim = (img[0].shape[1], img[0].shape[0])
            print('resizing image')
            dim = (i_size[1],i_size[0])
            img = cv2.resize(img[0], dim, interpolation = cv2.INTER_AREA) 
            resized = True    
       
        for shear_sys in tqdm(shearlet_systems):   
            for detect_param in all_detec_combs:
                if ridges:
                    #print(detect_param[0], detect_param[1])
                    features, orientations = shear_sys.detect(img[0], 
                                                              min_contrast=detect_param[0],  
                                                              offset = detect_param[1], 
                                                              pivoting_scales= pivoting_scales,
                                                              positive_only=positive, 
                                                              negative_only=negative
                                                              )
                else:
                    features, orientations = shear_sys.detect(img[0], 
                                                              min_contrast=detect_param[0],  
                                                              offset = detect_param[1], 
                                                              pivoting_scales= pivoting_scales
                                                              )
                Msum = np.sum(features)
                has_nan = np.isnan(Msum)
                if (has_nan):
                    print("NaN in array, check parameter combination.")
                else:
                   
                    if resized:
                        features = cv2.resize(features, old_dim, interpolation = cv2.INTER_AREA) 
                
                    detected = np.add(detected, features)
        norm = np.zeros(img[0].shape, np.double)
        normalized = cv2.normalize(detected, norm, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)          
        feature_img.append(normalized)
    return(feature_img)