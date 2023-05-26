# -*- coding: utf-8 -*-
"""
*********************************************************************
*				DO NOT MODIFY THIS HEADER					        *
*					FRACG - FRACture Graph					        *
*				Network analysis and meshing software		        *
*															        *
*						(c) 2022 CSIRO							    *
*GNU General Public Licence version 3 (GPLv3) with CSIRO Disclaimer	*
*																    *
*						Prepared by CSIRO						    *
*																    *
*					See license for full restrictions 			    *
*********************************************************************
@author: kel321
"""
import sys
import os
import cv2
import time
import numpy as np
import itertools
import skimage as sk
from PIL import Image
from PIL.ExifTags import TAGS

from osgeo import gdal,osr, ogr, gdal_array
from sknw import build_sknw
from coshrem.shearletsystem import EdgeSystem, RidgeSystem
from coshrem.util.image import mask, thin_mask#, curvature_rgb
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

def ReadMetaData(name):
    lis = []
    img = None
    try:
        img = Image.open(name)
    except:
        print('cannot retrieve image metadata')
    if img:
        exifdata = img.getexif()
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            if isinstance(data, bytes):
                data = data.decode().strip('\x00')   
            lis.append( (f'{tag}', f'{data}') )
    return(lis)

def Project2WGS84(pointX, pointY, pointZ, inputEPSG):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(pointX, pointY, pointZ)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(4326) #this is wgs84    
    coordTransform = osr.CoordinateTransformation(inputEPSG, outSpatialRef)
    point.Transform(coordTransform)
    X = point.GetX()
    Y = point.GetY()
    Z = point.GetZ()
    return(X, Y, Z)

def remap(array, noData):
    MIN = 0
    if noData:
        m = np.ma.masked_array(array, mask=(array==noData))
        MIN = m.min()
    else:
        MIN = array.min()
    if MIN >= 0 and array.max() <=1:
        array = array * 255
    if MIN < 0 or array.max() > 255:
        oldrange = array.max() - MIN
        if oldrange != 0:
            for i, d in enumerate(array):
                array[i] = (((d - MIN) * 255) / oldrange) 
        else:
            array.fill(0)
    return(array.astype(np.uint8))

#==============================================================================
def PrepareImages(Tools):
    Tools.RAW_IMG = []
    Tools.DATA = []
    if (len(Tools.FILE) > 0):
        ImgList = []
        hist_list = []
        proc_hist_list =[]
        for i, img in enumerate(Tools.FILE):
            dataset = None
            res = isinstance(img, str)
            if (res):
                dataset = gdal.Open(img, gdal.GA_ReadOnly)
                Tools.IMGMETA = ReadMetaData(os.path.abspath(img))
            else:
                print('cannot resolve filename', img)
            if dataset:
                if dataset.GetProjection():
                    if dataset.GetGeoTransform():
                        #Get the extend of the raster
                        proj = osr.SpatialReference(wkt=dataset.GetProjection())
                        ulx, xres, xskew, uly, yskew, yres  = dataset.GetGeoTransform()
                        newX, newY, _ = Project2WGS84(uly, ulx, 0.0, proj)
                        lrx = newX + (dataset.RasterXSize * xres)
                        lry = newY + (dataset.RasterYSize * yres)   
                        Tools.EXTEND = ((newX, newY), (lrx, lry) )
                        Tools.PROJ = dataset.GetProjection()
                        Tools.GEOT = dataset.GetGeoTransform()
                        noData = dataset.GetRasterBand(1).GetNoDataValue()
                        print("processing geotagged images...")
                        print("with ", dataset.RasterCount, " channel(s)")
                        bands = [dataset.GetRasterBand(k + 1) for k in range(dataset.RasterCount)]
  
                        driver = gdal.GetDriverByName('MEM')
                        dst_ds = driver.Create( '', dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount, gdal.GDT_Byte  )
                        dst_ds.SetProjection(dataset.GetProjectionRef())  
                        dst_ds.SetGeoTransform( dataset.GetGeoTransform() )

                        for i, b in enumerate(bands):
                            d = remap( b.ReadAsArray(), b.GetNoDataValue() )
                            dst_ds.GetRasterBand(i+1).WriteArray(d)
                        bands = [dst_ds.GetRasterBand(k + 1) for k in range(dst_ds.RasterCount)]
                        
                        for i, b in enumerate(bands):  
                            stats = b.GetStatistics(True, True)
                            print("band ",i+1, ": min: ", stats[0], "max: ", stats[1])                            
                            B = b.ReadAsArray()
                            #hist = cv2.calcHist([B],[0],None, [int(stats[1]) - int(stats[0]) +1], [stats[0], stats[1]])
                            hist = cv2.calcHist([B],[0],None, [255], (0,255))
                            hist_list.append(hist)
                        Tools.RAW_IMG.append(dst_ds)    #keep intial raster data
                    dataset = None
                else:
                    noData = None
                    print("processing non-geotagged images...")
                    rgb = cv2.imread(img, cv2.IMREAD_UNCHANGED) 
                    rgb = rgb.astype(np.uint8)
                    s = rgb.shape
                    if len(s) == 2:
                        b = 1
                    if len(s) > 2:
                        b = s[2]
                    print("with ", b, " channels")
                    
                    GeoT = np.zeros(6)
                    GeoT[0] = 0
                    GeoT[1] = 1
                    GeoT[2] = 0
                    GeoT[3] = 0
                    GeoT[4] = 0
                    GeoT[5] = -1
                    GeoT = tuple(GeoT)
                    
                    drv = gdal.GetDriverByName( 'MEM' )
                    DataType = gdal_array.NumericTypeCodeToGDALTypeCode(rgb.dtype)
                    dst_ds = drv.Create( '', s[1], s[0], b, DataType  )
                    dst_ds.SetProjection( '' )  
                    dst_ds.SetGeoTransform( GeoT )
                    
                    sp = cv2.split(rgb)
                    for i, b in enumerate(sp):   
                        m = b.min(axis=(0, 1))
                        M = b.max(axis=(0,1))
                        print("Channel: ", i+1, ": min: ", m, "max: ", M)
                        hist = cv2.calcHist([b],[0], None, [int((M-m)+1)], [m, M])
                        hist_list.append(hist)
                        dst_ds.GetRasterBand(i+1).WriteArray(b)
                    Tools.RAW_IMG.append(dst_ds)
                    dataset = None
                    
            #start the processing   
            print('preparing image')  
            if (dst_ds):      
                bands = dst_ds.RasterCount        
                if (Tools.RESIZE and Tools.PERCE != 100):
                    print('resizing')
                    #If resize enabled but percentage set to 100, calculate the ratio automatically
                    #based on a maximum dimension of MAXDIM
                    '''
                    if Tools.PERCE != 100:
                        #Auto resize calc
                        largest = max(dst_ds.RasterXSize, dst_ds.RasterYSize)
                        Tools.PERCE = 100 * Tools.MAXDIM // largest
                        print(f"Auto-resize to: {Tools.PERCE}%")
                    '''
                    #TODO: double check this!!!!!
                    width = int(dst_ds.RasterXSize * Tools.PERCE / 100)
                    height = int(dst_ds.RasterYSize * Tools.PERCE / 100)
                    drv = gdal.GetDriverByName( 'MEM' )
                    reproj = drv.Create( '', width, height, dst_ds.RasterCount, gdal.GDT_Byte )
                    reproj.SetProjection(dst_ds.GetProjectionRef())  
                    geoT = list( dst_ds.GetGeoTransform() )
                    geoT[1] = geoT[1] / (Tools.PERCE / 100)
                    geoT[5] = geoT[5] / (Tools.PERCE / 100)
                    geoT = tuple ( geoT )
                    reproj.SetGeoTransform( geoT )
                    gdal.ReprojectImage( dst_ds, reproj)
                    dst_ds = reproj
                    reproj = None
                rgb = dst_ds.ReadAsArray().astype("uint8")
                
                if (rgb.ndim) == 3:
                    rgb = np.swapaxes(rgb,0,2)
                    rgb = np.swapaxes(rgb,0,1)    
        
                if (bands > 2):
                    if (Tools. DETAIL): 
                        rgb = cv2.detailEnhance(rgb, sigma_s = Tools.SIG_S, sigma_r = Tools.SIG_R)
            
                if (Tools.GAMMA):
                    invGamma = 1.0 / Tools.GAM_C
                    table = np.array([((i / 255.0) ** invGamma) * 255
                                      for i in np.arange(0, 256)]).astype("uint8")
                    rgb = cv2.LUT(rgb, table)
                   
                if (bands > 2): 
                    if (Tools.WHITE):      
                        if (dst_ds.RasterCount == 4):
                            print('converting 4 channel to 3 channel image for white balanceing')
                            rgb = rgb[:, :, :3]
                        wb = cv2.xphoto.createGrayworldWB()
                        wb.setSaturationThreshold( Tools.W_THR ) 
                        rgb  = wb.balanceWhite( rgb )
        
                #create single bands raster 
                drv = gdal.GetDriverByName( 'MEM' )
                data = drv.Create( '', dst_ds.RasterXSize, dst_ds.RasterYSize, 1,  gdal.GDT_Byte  )
                data.SetProjection( dst_ds.GetProjectionRef() )  
                data.SetGeoTransform( dst_ds.GetGeoTransform() )
                
                if (noData):
                    data.GetRasterBand(1).SetNoDataValue(noData)
                
                if (bands == 1):
                    data.GetRasterBand(1).WriteArray( rgb )
                                        
                if (bands == 3):
                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    data.GetRasterBand(1).WriteArray( gray )
                    
                if (bands == 4):
                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)
                    data.GetRasterBand(1).WriteArray( gray )
                    
                elif (bands == 2 or bands > 4 ):
                    print(len(bands)," channel images are not supported")   
                    
                dst_ds = None
                ImgList.append(data)  
        
                stats = data.GetRasterBand(1).GetStatistics(True, True)
                print(stats)
                print("gray " ": min: ", stats[0], "max: ", stats[1])
                proc_hist_list.append( cv2.calcHist([data.GetRasterBand(1).ReadAsArray()],[0],None, [int(stats[1]) - int(stats[0]) +1], [stats[0], stats[1]]))
            Tools.DATA = ImgList
    return hist_list, proc_hist_list

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
def ReadImage(Tools):
    if (len(Tools.DATA) > 0):
        ImgList = []
        Tools.DATA2 = []
        for i, img in enumerate(Tools.DATA):
            
            driver = gdal.GetDriverByName("MEM")
            out = driver.CreateCopy('', img, strict=0) 
            
            gray = img.ReadAsArray()
            
            if ( Tools.HISTEQ ):
                arr = np.uint8(gray)
                gray = cv2.equalizeHist(arr)

            if ( Tools.GAUSBL ):
                gray = cv2.GaussianBlur(gray,(5,5),0)
           
                
            if (Tools.SHARPE):
                kernel = np.array([[-1,-1,-1], 
                                   [-1, 9,-1], 
                                   [-1,-1,-1]])
                gray = cv2.filter2D(gray, -1, kernel)
         
            if ( Tools.EDGE ):
                m_filter = np.array([[0,0,-1,0,0],
                                     [ 0,-1,-2,-1,0],
                                     [-1,-2,16,-2,-1],
                                     [0,-1,-2,-1,0],
                                     [0,0,-1,0,0]])
                gray = cv2.filter2D(gray, -1, m_filter)   

            if ( Tools.SOBEL ):
                x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
                y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
                gray =  (0.5*x) + (0.5*y) 
                
            if (Tools.INVERT):
                gray = cv2.bitwise_not(gray)
            
            out.GetRasterBand(1).WriteArray( gray )
            ImgList.append(out)
        Tools.DATA2 = ImgList

'''
check consient size of images and retun dimension of smallest image
args = list(str)
'''
def ImgSizes(images):
    s = []
    res = []
    for img in images:
        img = img.ReadAsArray(0)
        s.append(img.shape)
    [res.append(i) for i in s if i not in res]
    return(res)


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
def GenerateSystems(Tools):  
    Tools.R_SYS = []
    Tools.E_SYS = []
    i_size = ImgSizes(Tools.DATA3)
    wavelet_eff_supp  = SplitInput(Tools.WAVEEF, False)
    gaussian_eff_supp = SplitInput(Tools.GAUSEF, False)
    scales_per_octave = SplitInput(Tools.SCALES, True)
    shear_level       = SplitInput(Tools.SHEARL, True)
    ALPHA             = SplitInput(Tools.ALPHA, False)
    OCTAVES           = SplitInput(Tools.OCTAVE, False)

    t = time.time()
    

    all_sys_params = [wavelet_eff_supp, gaussian_eff_supp, scales_per_octave, shear_level, ALPHA, OCTAVES]
    all_sys_combs = list(itertools.product(*all_sys_params)) 
    print("generating ", len(all_sys_combs), " systems. For ", len(i_size), " different sizes.")
    for i, size in enumerate(i_size):
        print("system size: ", size)
        params = []
        for param in all_sys_combs:
            params.append([size, param[0], param[1], param[2],param[3],param[4],param[5]])
        iter_param = zip(params)
        print("CPUs: ",os.cpu_count())
        
        mw = os.cpu_count()
        
        if (len(all_sys_combs) < os.cpu_count()):
            mw = len(all_sys_combs)
            
        mw = 5   
        if (Tools.RIDGES):
            print(" ridge systems")
            Tools.R_SYS.append([])
            with ProcessPoolExecutor(max_workers = mw) as executor:
                for r in executor.map(GetRidgeSys, iter_param):
                    #shearlet_systems.append(r) 
                    Tools.R_SYS[i].append(r) 
                    
        if (Tools.EDGES):
            print(" edge systems")
            Tools.E_SYS.append([])
            with ProcessPoolExecutor(max_workers = mw) as executor:
                for r in executor.map(GetEdgeSys, iter_param):
                    #shearlet_systems.append(r) 
                    Tools.E_SYS[i].append(r) 
    elapsed = time.time() - t
    print(" done in ", elapsed, "s")   

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
    ridges   = pp[0][7]
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
        o_ret = np.zeros(img[0].shape, np.double)
    else:        
        f_ret = features
        o_ret = orientations
    return(f_ret, o_ret)

#Detect features in image
def DetectFeatures(Tools): 
    Tools.FEATURES = []
    min_contrast    = SplitInput(Tools.MINCON, False)
    offset          = SplitInput(Tools.OFFSET, False)
    pivoting_scales = Tools.PIVOTS
    positive        = Tools.POSITV
    negative        = Tools.NEGATI
    
    if (Tools.RIDGES):
        print('detecting features with ', len(Tools.R_SYS[0]), " systems.")
    t = time.time()

    # offset = CheckDetectionParams(offset)
    all_detec_params = [min_contrast, offset]
    all_detec_combs = list(itertools.product(*all_detec_params ))
    print(len(all_detec_combs), " detection combinations.")
    for i, img in enumerate(Tools.DATA3):    
         ds = img 
         img = img.GetRasterBand(1).ReadAsArray()
         detected = np.zeros(img.shape)#, np.double)  
         func_params = []    
         drv = gdal.GetDriverByName( 'MEM' )
         data = drv.Create( '', ds.RasterXSize, ds.RasterYSize, 1,  gdal.GDT_Float64 )
         data.SetProjection( ds.GetProjectionRef() )  
         data.SetGeoTransform( ds.GetGeoTransform() )
         ds = None   
         
         print("CPUs: ",os.cpu_count())
         mw = os.cpu_count()
         if (Tools.RIDGES):
             print('detecting ridges ', i+1, '/', len(Tools.DATA3))
             ridges = True
             for detect in all_detec_combs:
                 if (len(Tools.R_SYS) > 1):
                     cur_sysytems = Tools.R_SYS[i]
                 else:
                     cur_sysytems = Tools.R_SYS[0]
                 for shear_sys in cur_sysytems:   
                     func_params.append( (shear_sys, img, detect[0], detect[1], pivoting_scales, negative, positive, ridges) )    

             fp = zip(func_params)
             if (len(all_detec_combs) < os.cpu_count()):
                 mw = len(all_detec_combs)   
                
             
             with ProcessPoolExecutor(max_workers = mw) as executor:
                 for r, o in executor.map(Detect, fp): 
                     thinned_f = mask(r, thin_mask(r))
                     if not np.isnan(thinned_f.any()):
                         detected = np.add(detected, thinned_f)
                     else:
                        print('founnd NaN')
                     
         if (Tools.EDGES):
            print('detecting edges ', i+1, '/', len(Tools.DATA3))
            ridges = False
            for detect in all_detec_combs:
                if (len(Tools.R_SYS) > 1):
                     cur_sysytems = Tools.E_SYS[i]
                else:
                    cur_sysytems = Tools.E_SYS[0]
                for shear_sys in cur_sysytems:   
                    func_params.append( (shear_sys, img, detect[0], detect[1], pivoting_scales, negative, positive, ridges) ) 

            fp = zip(func_params)
            if (len(all_detec_combs) < os.cpu_count()):
                mw = len(all_detec_combs)   
                
            mw = 5
            with ProcessPoolExecutor(max_workers = mw) as executor:
                for r, o in executor.map(Detect, fp): 
                    thinned_f = mask(r, thin_mask(r))
                    if not np.isnan(thinned_f.any()):
                         detected = np.add(detected, thinned_f)
                    else:
                        print('founnd NaN')

         norm = np.zeros(img[0].shape, np.double)
         normalized = cv2.normalize(detected, norm, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)          
         data.GetRasterBand(1).WriteArray(normalized)
         Tools.FEATURES.append(data)
         data = None
    elapsed = time.time() - t
    print(" done in ", elapsed, "s")  


#Enhancing edge/ridge ensembles------------------------------------------------
'''


'''
def CleanIntensityMap(img):
    # define the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # opening the image
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
    img =  img - opening 
    p1, p99 = np.percentile(img, (1, 99))
    J = sk.exposure.rescale_intensity(img, in_range=(p1, p99))
    cleaned = sk.morphology.remove_small_objects(J, min_size=10, connectivity=2)     
    return(cleaned)

def SigmoidNonlinearity(image):
    ridges_norm_sig = np.zeros(image.shape, np.double)
    w,h = image.shape
    
    for i in range(w):
        for j in range(h):
            if image[i][j] != 0:
                ridges_norm_sig[i][j] = 1 / (1 + np.exp((-1)*image[i][j]))
    return(ridges_norm_sig)

def EnhanceEnsemble(Tools):
    enhanced_images  = []
    Tools.E_FEATURES = []
    for i, img in enumerate(Tools.FEATURES):  
        drv = gdal.GetDriverByName('MEM')   
        enh_feat = drv.Create( '', img.RasterXSize, img.RasterYSize, 1,  gdal.GDT_Int16 )
        enh_feat.SetProjection( img.GetProjectionRef() )  
        enh_feat.SetGeoTransform( img.GetGeoTransform() )
        img = img.GetRasterBand(1).ReadAsArray() 
        adjusted = Threshholding(img, Tools.THRESH, Tools.KSIZE)  
        skeleton = CleanUp(adjusted, Tools.MINSI)
        enh_feat.GetRasterBand(1).WriteArray(skeleton)
        enhanced_images.append( enh_feat )
        img = None   
    Tools.E_FEATURES = enhanced_images

def Threshholding(image, thresh, ksize):   
    print("Imgae thresholding")
    print(" Threshhold: ", thresh)
    w,h = image.shape
    #e_kernel = np.ones((5,5),np.uint8)#kernel for erosion
    
    if (thresh > 0):
        image[image < thresh] = 0
        closing = image
       #image  = cv2.GaussianBlur(image,(5,5),0)# reduce noise
       # kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(ksize),int(ksize)))#elliptic kernel 
       ## closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #closing small holes
    else:
        closing = image
    #erosion = cv2.erode(opening ,e_kernel,iterations = 1)#thinning the edges/ridges
    norm = np.zeros((w,h))
    final = cv2.normalize(closing, norm, 0, 255, cv2.NORM_MINMAX).astype("uint8")# normalize and convert to 8bit 
    return( final )

def CleanUp(image, m_size):
    minV = np.min(image[np.nonzero(image)])
    maxV = np.max(image)
    thresh = cv2.threshold(image, minV, maxV, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, 2)
    sizes = stats[1:, -1]; 
    nb_components = nb_components - 1
    img2 = output
    print("removing patches smaller than ", m_size)
    for i in range(0, nb_components):   #TODO: This is very slow
        if sizes[i] <= m_size:
            img2[output == i + 1] = 0
    img2[img2 > 0] = 255  
    print("skeletonizing")
    thinned = cv2.ximgproc.thinning(img2.astype("uint8"))        
    return(thinned)


def ConvertCoodinates(pt, dataset):
    if (dataset):
        transform = dataset.GetGeoTransform()
        X = float(transform[0] + int(pt[0]) * transform[1] )
        Y = float(transform[3] + (int(pt[1]) * transform[5] ))
        point = (X,Y)
        return(point)
    else:
        point = (pt[0],-pt[1])
        return(point)

def BuildSHP(img_list, filename, tolerance):
    shp = []
    for img in img_list:
        data = img.GetRasterBand(1).ReadAsArray()
        g = build_sknw(data, multi=False, iso=False, ring=True, full=True)
        shp.append( (g, img) )
    cur_dir = os.getcwd()
    for i, lines in enumerate(shp):     
        path = os.path.join(cur_dir, str(filename) + "_" + str(i) + ".shp")
        WritePolyline2SHP(lines[0], lines[1], path, tolerance) 
        print("written ", path)
        return(path)

def WritePolyline2SHP(graph, dataset, outputFile, tolerance): 
    #first convert the graph edges into a list of numpy arrays lists
    poly_lines = []
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']      
        poly_lines.append([ps[:,0], ps[:,1]])        
    #get the driver for ESRI shapefile
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if drv is None:
        print ("driver not available (ESRI Shapefile).")
        sys.exit( 1 )
        
   #  Create the shp-file (or open it if it exists)
    ds = drv.CreateDataSource(outputFile)
    if ds is None:
        print ("Creation of output file failed. Trying to open file...\n")
        ds = ogr.Open(outputFile, 1) # 0 means read-only. 1 means writeable.
        if ds is None:
            print ("Creation/opening of ", outputFile," failed.\n")
            sys.exit( 1 )          
# get layer and check if reference is given (write non-georeferenced shp if not defined)     
    lyr = ds.GetLayer(0)
    if lyr is None:
        if dataset:
            refWKT = dataset.GetProjection()
            SpatialRef = osr.SpatialReference()  # makes an empty spatial ref object
            SpatialRef.ImportFromWkt(refWKT) 
            lyr = ds.CreateLayer( "fitted_polylines", SpatialRef, ogr.wkbLineString)  
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
        for p in point_list:           
            POINT = (p[1], p[0])
            pt = ConvertCoodinates(POINT, dataset)      
            line.AddPoint(pt[0], pt[1])  
        simpleLine = line.Simplify(tolerance)   
        feat.SetGeometry(simpleLine)
        lyr.CreateFeature(feat) 
        feat.Destroy()
    ds = None

# +
def WritePoints2SHP(graph, dataset, outputFile, tolerance): 
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
        ds = ogr.Open(outputFile, 1) # 0 means read-only. 1 means writeable.
        if ds is None:
            print ("Creation of output file failed.\n")
            sys.exit( 1 )
            
# get layer and check if reference is given (write non-georeferenced shp if not defined)     
    lyr = ds.GetLayer(0)
    if lyr is None:
        if dataset:
            refWKT = dataset.GetProjection()
            SpatialRef = osr.SpatialReference()  # makes an empty spatial ref object
            SpatialRef.ImportFromWkt(refWKT) 
            lyr = ds.CreateLayer( "graph_vertices", SpatialRef, ogr.wkbPoint)  
        else:
            lyr = ds.CreateLayer( "graph_vertices", None, ogr.wkbPoint)
        if lyr is None:
            print ("Layer creation failed.\n")
            sys.exit( 1 )    
    lyr.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    lyr.CreateField(ogr.FieldDefn('degree', ogr.OFTInteger))
    
#loop though the points and create lines in the layer
    for i, p in enumerate(points): 
        if degrees[i] > 0:
            feat = ogr.Feature( lyr.GetLayerDefn())
            feat.SetField( "id", i )
            feat.SetField( "degree", degrees[i] )
            point = ogr.Geometry(ogr.wkbPoint)
            POINT = (p[1], p[0])
            pt = ConvertCoodinates(POINT, dataset)
            point.AddPoint(float(pt[0]), float(pt[1]))     
            feat.SetGeometry(point)
            lyr.CreateFeature(feat) 
            feat.Destroy()
    ds = None
    
def DynamicRangeCompression(img_list, c = 40):
    DRC = []
    
    def _dr1(img1):
        img1[img1 == 255] = 254
        img1=np.log(img1+ 1)
        return img1

    def _dr(frame, c):
        return (c * frame).astype(np.uint8)

    def chipka(bdr, gdr, rdr, img):
        q = []
        m, n, _ = img.shape
        for i, j, k in zip(bdr, gdr, rdr):
            q.append(list(zip(i, j, k)))
        return np.array(q).astype(np.uint8)
     
    for img in img_list:
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        bdr1, gdr1, rdr1 = map(lambda x: _dr1(x), (b, g, r))
 
        bdr, gdr, rdr = map(lambda x: _dr(x, c), (bdr1, gdr1, rdr1))
        res = chipka(bdr, gdr, rdr, img)
        DRC.append(res)
    return(DRC)

#------------------------------------------------------------------------------

