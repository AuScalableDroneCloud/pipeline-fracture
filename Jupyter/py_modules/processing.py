# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:28:28 2022

@author: kel321
"""
import sys
import cv2
import time
import numpy as np
import itertools
from osgeo import gdal
from skimage.morphology import skeletonize, binary_closing
from coshrem.shearletsystem import EdgeSystem, RidgeSystem
from concurrent.futures import ProcessPoolExecutor

#==============================================================================
'''
split input from text boxes 
args
text = str
isInt = bool
'''
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

'''
Check alpha parameter is in approriate range
args
alpha = float
'''
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

'''
Check offset parameters is in approriate range
args
offset = float    
'''
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

'''
reading the image list:
If image has georefereincing 
return Numpy array of first raster band and the OGR dataset
If no georeferencing information are availabe
return Numpy array of greyscale image and None
args
img_list = list(str)
histEq = bool -> apply histogramm equalization
gaussBl = bool -> apply Gaussian blur 
'''
def ReadImage(img_list, histEq = False, gaussBl = False, sharpen = False, edge = False, sobel = False, invert = False):
    if (len(img_list) > 0):
        ImgList = []
        cur_img = (None,None)
        for i, img in enumerate(img_list):
            dataset = None
            dataset = gdal.Open(img, gdal.GA_ReadOnly)
            if dataset:
                if dataset.GetProjection():
                    if dataset.GetGeoTransform():
                        gray = np.array(dataset.GetRasterBand(1).ReadAsArray())
                        if (histEq):
                            arr = np.uint8(gray)
                            gray = cv2.equalizeHist(arr)
                        if (gaussBl):
                            gray = cv2.GaussianBlur(gray,(5,5),0)
                        if (edge):
                            print('h')
                            m_filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
                            gray = cv2.filter2D(gray, -1, m_filter)
                        if (sobel):
                            x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
                            y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
                            gray = (0.5*x) + (0.5*y)
                        if (sharpen):
                            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                            gray = cv2.filter2D(gray, -1, kernel)
                        if (invert):
                            gray = cv2.bitwise_not(gray)
                        cur_img = (gray, dataset)
                else:
                    gray = cv2.imread(img, cv2.IMREAD_UNCHANGED)         
                    if gray.any() != None:
                        if len(gray.shape)==3:
                            gray  = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)  
                        if (histEq):
                            arr = np.uint8(gray)
                            gray = cv2.equalizeHist(arr)
                        if (gaussBl):
                            gray = cv2.GaussianBlur(gray,(5,5),0)
                        if (edge):
                            m_filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
                            gray = cv2.filter2D(gray, -1, m_filter)
                        if (sobel):
                            x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
                            y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
                            gray = (0.5*x) + (0.5*y)
                        if (sharpen):
                            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                            gray = cv2.filter2D(gray, -1, kernel)     
                        if (invert):
                            gray = cv2.bitwise_not(gray)                       
                        cur_img = (gray, None)
            else:
                print("Could not read image ", img)
                sys.exit()
            if (cur_img[0].any() != None):
                print('image size: ', cur_img[0].shape)
                ImgList.append(cur_img)    
            else:
                print("Could not read image ", img)
                sys.exit()
    else:
        print("no files selected")
        sys.exit()
    print('selected', len(ImgList), ' images')
    return(ImgList)

'''
write images as tif files:
If georeferencing information are given write a geotiff
If no georeferencing infromationare availabe write a simple tiff
args
img_list = list(tuple(Numpy array, dataset or None))
features = list(Numpy array)
filename = str
'''
def WriteImage(img_list, features, filename):
    assert len(img_list) == len(features),"number of images not equal to number of ridge/edge ensembles"
    for i, img in enumerate(img_list):
        outfile = filename + "_" + str(i) + ".tiff"
        if img[1] != None:
            dataset = img[1]          
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(outfile, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Int16)
            outdata.SetGeoTransform(dataset.GetGeoTransform())
            outdata.SetProjection(dataset.GetProjection())
            outdata.GetRasterBand(1).WriteArray(features[i])
            outdata.FlushCache() 
        else:
            status = cv2.imwrite(outfile, features[i])
            print("written non-georferenced image", status)

'''
check consient size of images and retun dimension of smallest image
args = list(str)
'''
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
 
#Generating shearlet sysyems and detecting features----------------------------
'''
Generate RidgeSystem from given list of parameters
args
params = zip(list())
'''
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

'''
Generate EdgeSystem from given list of parameters
args
params = zip(list())
'''
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

'''
Generate the complex shearlet systems based on lists.
All possible combinations are generated and the systems are then build usind mutiprocessing
args
i_size = (int,int) [the size of the image]
wavelet_eff_supp = list(float)
gaussian_eff_supp = list(float)
scales_per_octave = list(float)
shear_level = list(float)
ALPHA = list(float)
OCTAVES = list(float)
ridges = bool [this is used to switch between building ridge adn edge systems]
'''    
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
    mw = 10
    if (len(all_sys_combs) < 10):
        mw = len(all_sys_combs)
    with ProcessPoolExecutor(max_workers = mw) as executor:
        for r in executor.map(GetRidgeSys, iter_param):
            shearlet_systems.append(r) 
    elapsed = time.time() - t
    print(" done in ", elapsed, "s")   
    return(shearlet_systems)

'''
Detect the features in the image
'''
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
def DetectFeatures(img_list, i_size, shearlet_systems, min_contrast, offset, pivoting_scales, negative, positive, ridges): 
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
         mw = 10
         if (len(all_detec_combs) < 10):
             mw = len(all_detec_combs)
         with ProcessPoolExecutor(max_workers = mw) as executor:
             for r in executor.map(Detect, fp): 
                 detected = np.add(detected, r)
         norm = np.zeros(img[0].shape, np.double)
         normalized = cv2.normalize(detected, norm, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)          
         feature_img.append(normalized)
    elapsed = time.time() - t
    print(" done in ", elapsed, "s")  
    return(feature_img)

#Enhancing edge/ridge ensembles------------------------------------------------
'''

'''
def EnhanceEnsemble(features,  thresh = 0.01, ksize = 3, alpha = 1.5, connectivity = 3, min_size = 10):
    beta = 0
    enhanced_images = []
    for i, img in enumerate(features):
        adjusted = Threshholding(img, thresh, ksize, alpha, beta)
        skeleton = CleanUp(adjusted, connectivity, min_size)
        enhanced_images.append(skeleton)
    return(enhanced_images)
        
def SigmoidNonlinearity(image):
    ridges_norm_sig = np.zeros(image.shape, np.double)
    w,h = image.shape
    for i in range(w):
        for j in range(h):
            if image[i][j] != 0:
                ridges_norm_sig[i][j] = 1 / (1 + np.exp((-1)*image[i][j]))
    return(ridges_norm_sig)

def Threshholding(image, thresh, ksize, alpha, beta):   
    #img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary    
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
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape))
    for i in range(nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
    skeleton = binary_closing(skeletonize(img2))
    return(skeleton)

