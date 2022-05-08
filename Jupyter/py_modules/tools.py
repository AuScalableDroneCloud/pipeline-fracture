# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:01:32 2022

@author: kel321
"""

import cv2
import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import display

class Tools: 
    FILE = ['rgb.tiff']
    HISTEQ = False
    GAUSBL = False
    WAVEEF = ('25,50,150')
    GAUSEF = ('12,25,75')
    SCALES = ('2')
    SHEARL = ('3')
    ALPHA  = ('1')
    OCTAVE = ('3.5')
    RIDGES = True
    MINCON = ('5,10,25,50')
    OFFSET = ('1')
    PIVOTS = 'all'
    NEGATI = True
    POSITV = False 
    THRESH = 0.01  
    KSIZE = 3   
    CONTRA = 1.5  
    CONNE = 5 
    MINSI = 10
       
    def SelectFile(self):
        file = w.Dropdown(options=['rgb', 'dem', 'mag'],value='rgb',description='Imagetype:',disabled=False,)
        display(file)  
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.FILE = change['new']+'.tiff'
        file.observe(on_change)
              
    def SelectEnhancement(self):
        types = ["Histogram equalization", "Gaussian blur"]
        checkboxes = [w.Checkbox(value=False, description=t) for t in types]
        output = w.VBox(children=checkboxes)
        display(output)
        
        def on_tick_0(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.HISTEQ = change['new']
        def on_tick_1(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.GAUSBL = change['new']
                
        box0 = checkboxes[0]
        box1 = checkboxes[1]
        box0.observe(on_tick_0)
        box1.observe(on_tick_1)
        
    def SystemCombinations(self):
        style = {'description_width': 'initial'}
        waveletEffSupp = w.Text(value='25,50,150',placeholder='25,50,150',description='waveletEffSupp:',style=style, disabled=False)
        gaussianEffSupp = w.Text(value='12,25,75',placeholder='12,25,75',description='gaussianEffSupp:',style=style, disabled=False)
        scalesPerOctave = w.Text(value='2',placeholder='2',description='scalesPerOctave:',style=style, disabled=False)
        shearLevel = w.Text(value='3',placeholder='3',description='shearLevel:',style=style, disabled=False)
        alpha = w.Text(value='1',placeholder='1',description='alpha:',style=style, disabled=False)
        octaves = w.Text(value='3.5',placeholder='3.5',description='octaves:',style=style, disabled=False)
        ridges = w.Checkbox(True, description='Ridges')
        
        all_widgets =[waveletEffSupp, gaussianEffSupp, scalesPerOctave, shearLevel, alpha, octaves, ridges]
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
                self.SHEARL = change['new']              
        def change_octaves(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.OCTAVE= change['new']              
        def change_ridges(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.RIDGES= change['new']
                  
        all_widgets[0].observe(change_waveletEffSupp)
        all_widgets[1].observe(change_gaussianEffSupp)
        all_widgets[2].observe(change_scalesPerOctave)
        all_widgets[3].observe(change_shearLevel)
        all_widgets[4].observe(change_alpha) 
        all_widgets[5].observe(change_octaves)
        all_widgets[6].observe(change_ridges)
        
    def DetectionCombinations(self, RIDGES):
        style = {'description_width': 'initial'}
        min_contrast = w.Text(value='5,10,25,50',placeholder='5,10,25,50',description='minContrast:',style=style, disabled=False)
        offset = w.Text(value='1',placeholder='1,1.5',description='offset:',style=style, disabled=False)
        pivoting_scales = w.Dropdown(description='scalesUsedForPivotSearch',style=style, options=['all', 'highest', 'lowest'], value='all')
        if (RIDGES):
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
        all_widgets[3].observe(change_negative)
        all_widgets[4].observe(change_positive) 
        
    def Enhancement(self):
        style = {'description_width': 'initial'}
        thresh = w.FloatSlider(description='min pixel value',style=style, value=0.1 ,min=0, max=1, step=0.01,continuous_update=False)
        ksize  = w.IntSlider(description='kernel size',style=style, value=3 ,min=1, max=100, step=2,continuous_update=False)
        alpha  = w.FloatSlider(description='contrast',style=style, value=1.5, min=1, max=100, step=0.1,continuous_update=False)
        connec = w.IntSlider(description='connectivity',style=style, value=8 ,min=1, max=8, step=1,continuous_update=False)
        minsiz = w.IntSlider(description='min cluster size',style=style, value=10 ,min=1, max=1000, step=1,continuous_update=False)
        all_widgets = [thresh, ksize, alpha, connec, minsiz]
        output = w.VBox(children=all_widgets)
        display(output)
        
        def change_thresh(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.THRESH = change['new']       
        def change_ksize(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.KSIZE = change['new']             
        def change_alpha(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.CONTRA = change['new']            
        def change_connect(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.CONNE = change['new']          
        def change_min_size(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.MINSI = change['new']
                
        all_widgets[0].observe(change_thresh)
        all_widgets[1].observe(change_ksize)
        all_widgets[2].observe(change_alpha)
        all_widgets[3].observe(change_connect)
        all_widgets[4].observe(change_min_size) 
        
    def ShowImage(img_list):  
        for i, img in enumerate(img_list):
            f, ax1 = plt.subplots(nrows=1,figsize=(25,25))      
            ax1.imshow(img[0], cmap="gray"); 
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
    
    def ShowOverlay(img_list, features):  
        for i, img in enumerate(img_list):
            f, ax1 = plt.subplots(nrows=1,figsize=(25,25))
            overlay = cv2.addWeighted(img[0],0.005, features[i] ,0.99,0, dtype=cv2.CV_64F)       
            ax1.imshow(overlay, cmap="gray"); 
            ax1.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)    
    
    def ShowCompare(img_list1, img_list2):
        for i, img in enumerate(img_list1):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,25))
            ax1.imshow(img, cmap="gray")
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax2.imshow(img_list2[i])
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)