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

import os
import sys
import pathlib
import numpy as np
from osgeo import gdal
import ipywidgets as w
import matplotlib.pyplot as plt
from coshrem.util.image import overlay
from IPython.display import display
sys.path.append('src')
import gdal_retile

class Tools: 
    ASSETS = []         #webODM assets
    FILE = []           #files names of images -> list( str )
    
    #listst to store teh images during processing
    RAW_IMG = []        #unprocessed images including gdal obj -> list( (np.array, gdal_obj) ), note for non-reoreferenced data gdal_obj is None 
    DATA = []           #list containing processed data after image preparation -> list( (np.array, gdal_obj) ), note for non-reoreferenced data gdal_obj is None 
    DATA2 = []          #list containing processed data after image enhancement -> list( (np.array, gdal_obj) ), note for non-reoreferenced data gdal_obj is None 
    DATA3 = []          #list containing processed data after tiling -> list( (np.array, gdal_obj) ), note for non-reoreferenced data gdal_obj is None
    FEATURES = []       #list containing detected features -> np.array
    E_FEATURES = []
    
    #preparation
    RESIZE = True
    PERCE = 100
    MAXDIM = 2048 #Max image width or height for auto-resize
    SIG_S = 8
    SIG_R = 0.15
    GAMMA = False
    GAM_C = 1.0
    WHITE = False
    W_THR = 1.0
    
    #preprocessing
    HISTEQ = False
    GAUSBL = False
    SHARPE = False
    EDGE   = False
    SOBEL  = False
    DETAIL  = False
    INVERT = False
    
    TILED = False
    PIX = 500
    
    #system parameters
    EDGES  = False
    RIDGES = True
    WAVEEF = ('25,50,150')
    GAUSEF = ('12,25,75')
    SCALES = ('2')
    SHEARL = ('3')
    ALPHA  = ('1')
    OCTAVE = ('3.5')
    
    E_SYS = []
    R_SYS = []
    
    #detection
    POSITV = False 
    NEGATI = True
    MINCON = ('5,10,25,50')
    OFFSET = ('1')
    PIVOTS = 'all'
    
    #enhancement
    THRESH = 0.0 
    KSIZE = 3   
    MINSI = 1
    
    #names for metadata
    DAP = False
    JSON = False
    TERN = False
    USER = 'pet22a_wsi@DAPTst'
    PASSW = '^JwtGk^xM79D#_6&QXPCtkzG6M5kwC'
    SHP = ''
    GEOTIF = ''
    EXTEND = None
    PROJ = None
    GEOT = None
    J_NAME = 'sample'
    IMGMETA = []
        
    def Prepare4Processing(self):
        style = {'description_width': 'initial'}
        types = ["Resize", "DetailEnhance", "GammaCorrection", "WhiteBalance"]
        checkboxes = [w.Checkbox(value=False, description=t) for t in types]
        checkboxes[0].value = True #Resize default to True
        
        resize = w.IntText(value = self.PERCE, placeholder = self.PERCE, description='[%]:',style=style, disabled=False)
        sig_s = w.BoundedFloatText(value = self.SIG_R, placeholder = self.SIG_S, min = 0, max = 200, step=0.1, description='sig_s:',style=style, disabled=False)
        sig_r = w.BoundedFloatText(value = self.SIG_S, placeholder = self.SIG_R, min = 0, max = 1,   step=0.01,description='sig_r:',style=style, disabled=False)
        detail = w.HBox([sig_r, sig_s ])
        gamma = w.FloatText(value  = self.GAM_C, placeholder = self.GAM_C,  description='gamma:',style=style, disabled=False)
        white = w.FloatText(value  = self.W_THR, placeholder = self.W_THR,  description='thresh:',style=style, disabled=False)
                
        left_box  = w.VBox( [ checkboxes[0], checkboxes[1], checkboxes[2], checkboxes[3] ] )
        right_box = w.VBox( [resize, detail,  gamma, white] )
        
        output = w.HBox([left_box, right_box])
        display(output)

        def on_tick_0(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.RESIZE = change['new']
                
        def on_tick_1(change):
            if change['type'] == 'change' and change['name'] == 'value':
               self.DETAIL = change['new']
                
        def on_tick_2(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.GAMMA = change['new']
                          
        def on_tick_3(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.WHITE = change['new']   
        def change_size(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.PERCE = change['new']  
        def change_sig_S(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.SIG_S = change['new'] 
        def change_sig_R(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.SIG_R = change['new']          
        def change_GAM_C(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.GAM_C = change['new']       
        def change_W_THR(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.W_THR = change['new'] 
                
        checkboxes[0].observe(on_tick_0)
        checkboxes[1].observe(on_tick_1)
        checkboxes[2].observe(on_tick_2)
        checkboxes[3].observe(on_tick_3) 
        resize.observe(change_size)
        sig_r.observe(change_sig_S)
        sig_s.observe(change_sig_S)
        gamma.observe(change_GAM_C)
        white.observe(change_W_THR)      

    def PlotPreparedImg(self, hist, proc_hist):
        assert len(Tools.RAW_IMG) == len(Tools.DATA), "inconsistent image lists"
        for i, img in enumerate(Tools.DATA):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(25,25))
            raw = Tools.RAW_IMG[i].ReadAsArray()
            if (raw.ndim) == 3:
                raw = np.swapaxes(raw,0,2)
                raw = np.swapaxes(raw,0,1)  
            img = img.ReadAsArray()
            ax1.imshow(raw, cmap=plt.get_cmap('gray'))
            for h in hist:
                ax2.plot( h )
            ax3.imshow(img, cmap=plt.get_cmap('gray') )
            for h in proc_hist:
                ax4.plot( h )

    def SelectFile(self):
        file = w.Dropdown(options=['rgb', 'rgb_2', 'rgb_3', 'dem', 'mag'],value='rgb',description='Imagetype:',disabled=False,)
        display(file)  
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.FILE = ['img/examples/' + change['new']+'.tif']
        file.observe(on_change)
                     
    def CheckTemp():
        cur_dir = os.getcwd()
        path = os.path.join(cur_dir, 'temp')
        if os.path.isdir(path): 
            if (len(os.listdir(path)) > 0):
                for f in os.listdir(path):
                    try:
                        os.remove(os.path.join(path, f))
                    except:
                        print('cannot remove file ', f)        
        else:
            os.mkdir(path)
            print("Directory '% s' created" % path)
                  
    def GetFromTemp(self):
        self.DATA3 = []
        files  = [] 
        cur_dir = os.getcwd()
        path = os.path.join(cur_dir, 'temp')
        if os.path.isdir(path): 
            if (len(os.listdir(path)) > 0):
                for f in os.listdir(path):
                    files.append(os.path.join(path, f))
            else:
                print('temp folder is empty. Check pixel size for tiling!')
                
        if (len(files) > 0):
            for i, img in enumerate(files):
                 res = isinstance(img, str)
                 if (res):
                     dataset = gdal.Open(img, gdal.GA_ReadOnly)
                 else:
                     print('cannot resolve filename', img)
                 if dataset:
                     driver = gdal.GetDriverByName("MEM")
                     data = driver.CreateCopy('', dataset, strict=0) 
                     self.DATA3.append( data )
                     dataset = None
                           
    def Tile(self):  
        if (int(self.PIX) != 0):
            self.CheckTemp()
            cur_dir = os.getcwd()
            file = os.path.join(cur_dir, 'temp', 'temp_file.tif') 
            for dataset in self.DATA2:  
               i = dataset.ReadAsArray()
               driver = gdal.GetDriverByName("GTiff")
               outdata = driver.Create(file, dataset.RasterXSize, dataset.RasterYSize, 1, dataset.GetRasterBand(1).DataType )
               outdata.SetGeoTransform(dataset.GetGeoTransform())
               outdata.SetProjection(dataset.GetProjection())
               outdata.GetRasterBand(1).WriteArray( i )
               outdata.FlushCache() 
               outdata = None
               dataset = None
               cmd=(' ','-ps',str(self.PIX),str(self.PIX),'-targetDir','temp', file)
               gdal_retile.main(cmd)
               os.remove(file)
               self.GetFromTemp(self)
               self.TILED = True
        else:
            print('define a pixel size')
               
    def PixelSize(self):
         pixel = w.Text(value='500', placeholder='x y pixel', description='Pixel Size:', disabled=False)
         go = w.Button(description='GO!')
         out2 = w.Output()

         def on_change(change):
             if change['type'] == 'change' and change['name'] == 'value':
                 self.PIX = change['new']   
                 
         def go_tile(obj):
             with out2:
                 self.Tile(self)
                 print('done')
            
         pixel.observe(on_change)
         go.on_click(go_tile)
         
         tile = w.HBox([pixel, go])
         show2 = w.VBox([tile, out2]) 
         display(show2)     
      
    def TileImage(self, default=None):
        if (len(self.DATA2) == 1):
            Tools.CheckTemp()
            btn = w.Button(description='Tile image')
            bt2 = w.Button(description='Do not tile image')
            out = w.Output()
           
            def Tile(obj):
                with out:
                    print('Please select the desired tile size in pixels')
                    self.PixelSize(self)
    
            def NoTile(obj):
                with out:
                    print('processing with initial image size')
                    self.DATA3 = []
                    self.DATA3 = self.DATA2
    
            btn.on_click(Tile)
            bt2.on_click(NoTile)

            if default == 0:
                NoTile(self)
            elif default == 1:
                Tile(self)
            
            buttons = w.HBox([btn, bt2])
            show = w.VBox([buttons, out])
            display(show)
        else:
            print('Tiling image lists not implemented.')
            self.DATA3 = self.DATA2
       
    def MosaicTiles(self):
        images = []
        featur = []
        IMG = []
        FEA = []
        assert len(self.DATA3) == len(self.FEATURES), "inconsistent image lists"
        if (self.TILED):
            print("retiling, ", len(self.FEATURES), " images.")
            for i, img in enumerate(self.DATA3): 
                detected = Tools.FEATURES[i].ReadAsArray()
                driver = gdal.GetDriverByName("MEM")
                features = driver.Create('', img.RasterXSize, img.RasterYSize, 1, gdal.GDT_Float64)
                outdata  = driver.Create('', img.RasterXSize, img.RasterYSize, 1, img.GetRasterBand(1).DataType)
                outdata.SetGeoTransform(img.GetGeoTransform())
                outdata.SetProjection(img.GetProjection())
                features.SetGeoTransform(img.GetGeoTransform())
                features.SetProjection(img.GetProjection())
                outdata.GetRasterBand(1).WriteArray(img.GetRasterBand(1).ReadAsArray() )   
                features.GetRasterBand(1).WriteArray(detected)
                images.append( outdata  )
                featur.append( features )   
            I = gdal.Warp('', images, format="MEM", options=["COMPRESS=LZW", "TILED=YES"]) 
            F = gdal.Warp('', featur, format="MEM", options=["COMPRESS=LZW", "TILED=YES"]) 
            IMG.append(I)
            FEA.append(F)
        else:
            for i, feat in enumerate(self.FEATURES):
                detected = feat.ReadAsArray()
                img = self.DATA3[i]
                driver = gdal.GetDriverByName("MEM")
                features = driver.Create('', img.RasterXSize, img.RasterYSize, 1, gdal.GDT_Float64)
                features.SetGeoTransform(img.GetGeoTransform())
                features.SetProjection(img.GetProjection())
                features.GetRasterBand(1).WriteArray(detected)
                IMG.append( img )
                FEA.append( features)
        self.DATA = IMG 
        self.FEATURES = FEA   

        if (self.RESIZE):
            print("converting back to original size")
            assert len(self.RAW_IMG) == len(self.FEATURES), "inconsistent image lists"
           
            for i, img in enumerate(self.RAW_IMG): 
                dataset = self.FEATURES[i]
                drv = gdal.GetDriverByName( 'MEM' )
                dst_ds = drv.Create( '', img.RasterXSize, img.RasterYSize, 1, gdal.GDT_Float64  )
                dst_ds.SetGeoTransform( img.GetGeoTransform() )
                dst_ds.SetProjection(img.GetProjectionRef())  
                dst_ds.GetRasterBand(1).WriteArray(dataset.GetRasterBand(1).ReadAsArray()) 
                e = gdal.ReprojectImage( dataset, dst_ds, None,  None, gdal.GRA_NearestNeighbour ) 
                if e != 0:
                    print("Failed to reproject image")
                self.FEATURES[i] = (dst_ds)
                img = None
                dst_ds = None  
                dataset = None
                drv = None
   
    def SelectEnhancement(self):
        types = ["Histogram equalization", "Gaussian blur", "Sharpen", "Sobel", "Edge", "Invert"]
        checkboxes = [w.Checkbox(value=False, description=t) for t in types]
        output = w.VBox(children=checkboxes)
        display(output)
        
        def on_tick_0(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.HISTEQ = change['new']
        def on_tick_1(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.GAUSBL = change['new']
        def on_tick_2(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.SHARPE = change['new']
        def on_tick_3(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.SOBEL = change['new']
        def on_tick_4(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.EDGE = change['new']
        def on_tick_5(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.INVERT = change['new']
                
        box0 = checkboxes[0]
        box1 = checkboxes[1]
        box2 = checkboxes[2]
        box3 = checkboxes[3]
        box4 = checkboxes[4]
        box5 = checkboxes[5]

        box0.observe(on_tick_0)
        box1.observe(on_tick_1)
        box2.observe(on_tick_2)
        box3.observe(on_tick_3)
        box4.observe(on_tick_4)
        box5.observe(on_tick_5)

        
    def SystemCombinations(self):
        style = {'description_width': 'initial'}
        waveletEffSupp = w.Text(value='25,50,150',placeholder='25,50,150',description='waveletEffSupp:',style=style, disabled=False)
        gaussianEffSupp = w.Text(value='12,25,75',placeholder='12,25,75',description='gaussianEffSupp:',style=style, disabled=False)
        scalesPerOctave = w.Text(value='2',placeholder='2',description='scalesPerOctave:',style=style, disabled=False)
        shearLevel = w.Text(value='3',placeholder='3',description='shearLevel:',style=style, disabled=False)
        alpha = w.Text(value='1',placeholder='1',description='alpha:',style=style, disabled=False)
        octaves = w.Text(value='3.5',placeholder='3.5',description='octaves:',style=style, disabled=False)
        ridges = w.Checkbox(True, description='Ridges')
        edges  = w.Checkbox(False, description='Edges')
        
        all_widgets =[waveletEffSupp, gaussianEffSupp, scalesPerOctave, shearLevel, alpha, octaves, ridges, edges]
        output = w.VBox(children=all_widgets)
        display(output)
        
        def change_waveletEffSupp(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.WAVEEF = change['new']             
        def change_gaussianEffSupp(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.GAUSEF = change['new']           
        def change_scalesPerOctave(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.SCALES = change['new']          
        def change_shearLevel(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.SHEARL = change['new']
        def change_alpha(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.ALPHA = change['new']              
        def change_octaves(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.OCTAVE= change['new']              
        def change_ridges(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.RIDGES= change['new']
        def change_edges(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.EDGES= change['new']
                  
        all_widgets[0].observe(change_waveletEffSupp)
        all_widgets[1].observe(change_gaussianEffSupp)
        all_widgets[2].observe(change_scalesPerOctave)
        all_widgets[3].observe(change_shearLevel)
        all_widgets[4].observe(change_alpha) 
        all_widgets[5].observe(change_octaves)
        all_widgets[6].observe(change_ridges)
        all_widgets[7].observe(change_edges)
        
    def DetectionCombinations(self):
        style = {'description_width': 'initial'}
        min_contrast = w.Text(value='5,10,25,50',placeholder='5,10,25,50',description='minContrast:',style=style, disabled=False)
        offset = w.Text(value='1',placeholder='1,1.5',description='offset:',style=style, disabled=False)
        pivoting_scales = w.Dropdown(description='scalesUsedForPivotSearch',style=style, options=['all', 'highest', 'lowest'], value='all')
        if (self.RIDGES):
            negative = w.Checkbox(True, description='negative')
            positive = w.Checkbox(False, description='positive')
            all_widgets = [min_contrast, offset, pivoting_scales, negative, positive]
        else:
            all_widgets = [min_contrast, offset, pivoting_scales]            
        output = w.VBox(children=all_widgets)
        display(output)
        
        def change_min_contrast(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.MINCON = change['new']        
        def change_offset(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.OFFSET = change['new']                 
        def change_pivoting_scales(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.PIVOTS = change['new'] 
        def change_negative(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.NEGATI = change['new'] 
        def change_positive(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.POSITIVE = change['new'] 
                
        all_widgets[0].observe(change_min_contrast)
        all_widgets[1].observe(change_offset)
        all_widgets[2].observe(change_pivoting_scales)
        
        if (self.RIDGES):
            all_widgets[3].observe(change_negative)
            all_widgets[4].observe(change_positive) 
        
    def Enhancement(self):
        style = {'description_width': 'initial'}
        thresh = w.BoundedFloatText(value = self.THRESH, placeholder = self.THRESH, min = 0, max = 1, step=0.01, description='min pixel value:',style=style, disabled=False)
        minsiz = w.BoundedFloatText(value = self.MINSI, placeholder = self.MINSI, min = 1, max = 1000000, step=1, description='min cluster size:',style=style, disabled=False)
        all_widgets = [thresh, minsiz]
        output = w.VBox(children=all_widgets)
        display(output)
        
        def change_thresh(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.THRESH = change['new']       
                        
        def change_min_size(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.MINSI = change['new']
                
        all_widgets[0].observe(change_thresh)
        all_widgets[1].observe(change_min_size)
        
    def ShowImage(img_list):  
        for i, img in enumerate(img_list):
            img = img.ReadAsArray()
            f, ax1 = plt.subplots(nrows=1,figsize=(25,25))      
            ax1.imshow(img, cmap="gray")
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
      
    def ShowOverlay(self):  
        assert len(self.FEATURES) == len(self.RAW_IMG), "hmm"  
        for i, img in enumerate(self.RAW_IMG):
            feat = self.FEATURES[i].GetRasterBand(1).ReadAsArray()
            image = img.GetRasterBand(1).ReadAsArray()
            values = feat[feat!=0]
            m = np.min(values)
            M = np.max(values)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,25))
            ax1.imshow(overlay(image, feat) )
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)    
            ax2.plot(feat[0], color='k')
            ax2.set_ylim((m, M))

    def ShowCompare(self):
        assert len(self.FEATURES) == len(self.E_FEATURES), "hmm"  
        for i, feat in enumerate(self.FEATURES):
            img1 = feat.GetRasterBand(1).ReadAsArray() 
            img2 = self.E_FEATURES[i].GetRasterBand(1).ReadAsArray() 
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,25))
            ax1.imshow(img1, cmap="gray")
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax2.imshow(img2, cmap="gray_r")
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
         
    def SelectFilename(self, T):
        if (str(T) == 'tif'):
            Tools.GEOTIF = 'test'
        if (str(T) == 'shp'):
            Tools.SHP = 'test'
        if (str(T) == 'json'):
            Tools.J_NAME = 'test'
        filename = w.Text(value='test', placeholder='filename', description='Filename:', disabled=False)
        display(filename)
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                if (str(T) == 'tif'):
                    Tools.GEOTIF = change['new'] 
                if (str(T) == 'shp'):
                    Tools.SHP = change['new'] 
                if (str(T) == 'json'):
                    Tools.J_NAME = change['new'] 
        filename.observe(on_change)
           

        
    '''
    write images as tif files:
    If georeferencing information are given write a geotiff
    If no georeferencing infromationare availabe write a simple tiff
    args
    img_list = list(tuple(Numpy array, dataset or None))
    features = list(Numpy array)
    filename = str
    '''
    def WriteImage(self, img_list, filename): 
        print("Images will be written into ", os.getcwd())
        for i, img in enumerate(img_list):
            cur_dir = os.getcwd()
            path = os.path.join(cur_dir, filename + "_" + str(i) + ".tif")
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(path, img.RasterXSize, img.RasterYSize, 1, img.GetRasterBand(1).DataType)
            outdata.SetGeoTransform(img.GetGeoTransform())
            outdata.SetProjection(img.GetProjection())
            outdata.GetRasterBand(1).WriteArray(img.GetRasterBand(1).ReadAsArray() )
            outdata.FlushCache() 
            print("written image ", filename)
            Tools.GEOTIF = path
            outdata = None

#WEBODM_part-------------------------------------------------------------------
    def GetAssets(self, project, task):
        Tools.FILE = []
        assests = ['orthophoto.tif', 'dsm.tif']
        if not project: project = '622'
        if not task: task = 'b4b2382f-1946-4593-99c7-a01615d9ebd4'
    
        pathlib.Path(task).mkdir(parents=True, exist_ok=True)
        os.chdir(task)
        import asdc
        for i in assests:
            r = asdc.download_asset(i, project=project, task=task)
            if r and "orthophoto" in i or "dsm" in i:
                Tools.ASSETS.append( str(i) )     
                print('added', i, 'to downloaded assets.')  
        if (len(Tools.ASSETS) > 0):
            Tools.FILE.append( Tools.ASSETS[0])
        else:
            print('ERROR: Could not retrieve assets')
            sys.exit()
                
    def SelectAsset(self):     
        names = [] 
        for f in Tools.ASSETS:
            names.append(f)            
        file = w.Dropdown(options=names,description='Assets:', value='orthophoto.tif', disabled=False,)
        display(file)  
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                Tools.DATA = []
                for d in Tools.ASSETS:
                    if d == change['new']:
                        Tools.FILE = []
                        Tools.FILE.append( d )
        file.observe(on_change)

#CSIRO DAP---------------------------------------------------------------------
    def MetaData(self):
        types = ["JSON", "CSIRO DAP (test)", "TERN"]
        checkboxes = [w.Checkbox(value=False, description=t) for t in types]
                        
        box  = w.VBox( [ checkboxes[0], checkboxes[1], checkboxes[2] ] )
        output = w.HBox([box])
        display(output)

        def on_tick_0(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.JSON = change['new']
                
        def on_tick_1(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.DAP = change['new']
        
        def on_tick_2(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.TERN = change['new']
                
        checkboxes[0].observe(on_tick_0)
        checkboxes[1].observe(on_tick_1)
        checkboxes[2].observe(on_tick_2)
        
        '''
        if (self.DAP == True):
            print('Please give your credentials.')
            self.GetCredentials(self)
        '''

    def GetCredentials(self):
        user  = w.Text(value='', placeholder='', description='Username:', disabled=False)
        passw = w.Text(value='', placeholder='', description='Password:', disabled=False)
        all_widgets = [user, passw]
        output = w.VBox(children=all_widgets)
        display(output)
        
        def change_user(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.USER = change['new']       
                        
        def change_passw(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.PASSW = change['new']
                
        all_widgets[0].observe(change_user)
        all_widgets[1].observe(change_passw)
        

        

        
