import sys, os
import random
import unittest 
import numpy as np

sys.path.append("../src/") 
import processing
from tools import *

def CreateRandomList(size, mi, ma):
    rand_list=[]
    for i in range(size):
        rand_list.append(random.randint(mi,ma))
    rand_list = ','.join(map(str,rand_list))
    return(rand_list)

class test_asdc_sguc(unittest.TestCase):
    data = [f for f in os.listdir('test_data/') 
            if os.path.isfile(os.path.join('test_data/', f))]
    
    def rand_parameters(t = 0):
        if t == 0:
            print('creating system parameters')
        else:
            print('creating detection parameteres')
  
    def test_reading(self):

        for test in range(5):
            if test == 1:
                Tools.RESIZE = True
                Tools.PERCE = 90
                
            if test == 2:
                Tools.RESIZE = False
                Tools.DETAIL = True
                Tools.HISTEQ = True
                
            if test == 3:
                Tools.DETAIL = False
                Tools.GAMMA = True
                Tools.GAM_C = 0.9
                
            if test == 4:
                Tools.GAMMA = False
                Tools.WHITE = True
                Tools.W_THR = 0.9
                
            if test == 5:
                Tools.RESIZE = True
                Tools.PERCE = 10
                Tools.DETAIL = True
                Tools.GAMMA = True
                Tools.GAM_C = 0.8
                Tools.WHITE = True
                Tools.W_THR = 1.1
                Tools.HISTEQ = True
                Tools.GAUSBL = True
                Tools.SHARPE = True
                Tools.EDGE   = True
                Tools.SOBEL  = True
                Tools.DETAIL  = True
                Tools.INVERT = True

            for i, d in enumerate(self.data):
                 Tools.FILE = []
                 Tools.FILE.append('test_data/'+ d)
    
                 print(Tools.FILE)
                 hist, proc_hist = processing.PrepareImages(Tools)
                 processing.ReadImage(Tools) 
                 print(' ')

    def test_system_creation(self):
        dataset = gdal.Open('test_data/dem.tif', gdal.GA_ReadOnly)
        Tools.DATA3.append( dataset)
        for test in range (2):
            Tools.WAVEEF = CreateRandomList(5, 10, 500)
            Tools.GAUSEF = CreateRandomList(5, 10, 500)
            print('Wavelet Effective Support: ', Tools.WAVEEF)
            print('Gaussian Effective Support: ', Tools.GAUSEF)
            processing.GenerateSystems(Tools)
            processing.DetectFeatures(Tools)
        
    def test_filtering(self):
        processing.EnhanceEnsemble(Tools)
        
def main():
    unittest.main()

if __name__ == "__main__":
    main()
