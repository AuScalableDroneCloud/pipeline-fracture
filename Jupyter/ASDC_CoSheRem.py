# -*- coding: utf-8 -*-
"""
**********************************************************************
*				     DO NOT MODIFY THIS HEADER					     *
*					     ASDC - CoSheRem					         *
*	   Feature detection using the Complex Shearlet Transform		 *
*															         *
*						   (c) 2022 CSIRO							 *
* GNU General Public Licence version 3 (GPLv3) with CSIRO Disclaimer *
*																     *
*						  Prepared by CSIRO						     *
*																     *
*					See license for full restrictions 			     *
**********************************************************************
@author: kel321
"""

import os
import sys
import argparse
sys.path.append('src')
from tools import *
from processing import *

#split filenames at comma 
def create_list(names):
    split_names = names.split(",")
    for n in split_names:
        Tools.FILE.append(n)

class ASDC_CoSheRem:
    
    #required parameters
    filename = []
    outfile1 = None
    outfile2 = None  
        
    def __init__(self, filename, outfile1, outfile2,                    # filenames
                 percent, sig_r, sig_s,  gamma, wb,                     # image preparation parameters
                 histEq, gaussB, sharpen, edge, sobel, invert, 
                 edges, ridges, wave, gaus, scal, shea, alpha, octa,    # shearlet system parameters
                 positive, negative, m_con, offS,
                 thresh, minS):   
      
        create_list(filename)
        
        if (percent and percent != 100):
            print('resizing')
            Tools.RESIZE = True
            Tools.PERCE = percent
        if (sig_r != 0 and sig_s !=0):
            Tools.DETAIL = True
            Tools.SIG_S = sig_s
            Tools.SIG_R = sig_r
        if (gamma != 0):
            Tools.GAMMA = True
            Toosl.GAM_C = True
        if (wb != 0):
            Tools.WHITE = True
            
        Tools.HISTEQ = histEq
        Tools.GAUSBL = gaussB
        Tools.SHARPE = sharpen
        Tools.EDGE   = edge
        Tools.SOBEL  = sobel
        Tools.INVERT= invert
        
        #Shearlet system parameters
        if edges:
            print('edges on')
            Tools.EDGES = True
        if ridges:
            print('ridges on')
            Tools.RIDGES = True
        if not Tools.EDGES and not Tools.RIDGES:
            sys.exit('ERROR!')
        Tools.WAVEEF = wave
        Tools.GAUSEF = gaus
        Tools.SCALES = scal
        Tools.SHEARL = shea
        Tools.ALPHA  = alpha
        Tools.OCTAVE = octa
        
        #Detection parameters
        Tools.POSITV = positive
        Tools.NEGATI = negative
        Tools.MINCON = m_con
        Tools.OFFSET = offS
        Tools.PIVOTS = 'all'
        
        #Enhancemnet Parameters
        Tools.THRESH = thresh
        Tools.KSIZE = 3
        Tools.MINSI = minS
    
def Usage():
    print('Usage: ASDC_CoSheRem.py ')
    
def main(args):

    parser = argparse.ArgumentParser(description='ASDC_CoSheRem: Complex Sheralet edge/ridge measure')
    
    # Filenames
    parser.add_argument('--file', type=str, help='Required filename')
    parser.add_argument('--out1', type=str, help='Filename of processes raster')
    parser.add_argument('--out2', type=str, help='Filename of processes shp')
    
    #Optinal arguments
    parser.add_argument('--resize', type=int,   default=100, help='resize image in % of initial size')
    parser.add_argument('--sigR',   type=float, default=0.0, help='sigR for detail enahcement')
    parser.add_argument('--sigS',   type=float, default=0.0, help='sigS for detail enahcement')
    parser.add_argument('--gamma',  type=float, default=0.0, help='gamma correction factor')
    parser.add_argument('--wb',     type=float, default=0.0, help='white balance factor')
 
    # enahnce image switches
    parser.add_argument('--histEq',  action='store_true', help='histrogram equalization')
    parser.add_argument('--gaussB',  action='store_true', help='Gaussian blur')
    parser.add_argument('--sharpen', action='store_true', help='Sharpen mask')
    parser.add_argument('--edge',    action='store_true', help='Simple edges')
    parser.add_argument('--sobel',   action='store_true',  help='Image gradient')
    parser.add_argument('--invert',  action='store_true', help='Invert image')
    
    #Sytem Building parameters
    parser.add_argument('--edges',  action='store_true',help='switch on edgedetection')
    parser.add_argument('--ridges', action='store_true',help='switch on ridgedetection')
    parser.add_argument('--wave', type=str, default = ('50, 150'), help='Wavelet effective support')
    parser.add_argument('--gaus', type=str, default = ('25, 75'),  help='Gaussian effective support')
    parser.add_argument('--scal', type=str, default = ('2'),       help='Scales')
    parser.add_argument('--shea', type=str, default = ('3'),       help='Shearlet')
    parser.add_argument('--alph', type=str, default = ('1'),       help='Alpha')
    parser.add_argument('--octa', type=str, default = ('3.5'),     help='Octaves')
    
    #Detection Parameters
    parser.add_argument('--positive',  action='store_true', default = False,  help='positive ridges')
    parser.add_argument('--negative',  action='store_true', default = True, help='negative ridges')
    parser.add_argument('--minC', type=str, default = ('1, 5'), help='minimum contrast')
    parser.add_argument('--offS', type=str, default = ('1'), help='minimum contrast')
    
    #Enhancemnt Parameters
    
    parser.add_argument('--thresh',  type=float, default=0.0, help='feature threshold')
    parser.add_argument('--minS',  type=int, default=0.0, help='minimum cluster size')
    
    args = parser.parse_args()
    run = ASDC_CoSheRem (args.file, args.out1, args.out2, 
                         args.resize, args.sigR, args.sigS, args.gamma, args.wb,
                         args.histEq, args.gaussB, args.sharpen, args.edge, args.sobel, args.invert,
                         args.edges, args.ridges, args.wave, args.gaus, args.scal, args.shea, args.alph, args.octa,
                         args.positive, args.negative, args.minC, args.offS,
                         args.thresh, args.minS)

    Tools.E_SYS = []
    Tools.R_SYS = []
    hist, proc_hist = PrepareImages(Tools)  
    ReadImage(Tools)

    #Build Shearlet systems
    Tools.DATA3 = Tools.DATA2
    print('\n System aparameters: \n Wavelet effective suffort:', args.wave, '\n Gaussian effective support: ', args.gaus, '\n Scales per octave: ', args.scal, '\n Shearlevel: ', args.shea, '\n Alpha: ', args.alph, '\n Octaves: ', args.octa, '\n')
    GenerateSystems(Tools)  
    
    #Detect features
    print('Detection parameters:\n minimum contrast: ', args.minC, '\n offset: ', args.offS)
    DetectFeatures(Tools)
    Tools.MosaicTiles(Tools)
    Tools.WriteImage(Tools.FEATURES, args.out1)
    
    #Clean up and skeletonize
    print('Cleaning feature map:\n minimum pixel value: ', args.thresh, '\n minimum cluster size: ',args.minS)
    EnhanceEnsemble(Tools)
    BuildSHP(Tools.E_FEATURES, args.out2, 100)
    return(0)
   
if __name__ == '__main__':
    sys.exit(main(sys.argv))