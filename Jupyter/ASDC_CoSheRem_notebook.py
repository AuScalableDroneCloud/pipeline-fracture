# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ![logo-2.png](attachment:logo-2.png)
#
# ## <center>Structural Geology Use Case</center>
#
# Workflow for fracture detection with Complex Shearlet Transform based on: 
# https://github.com/rahulprabhakaran/Automatic-Fracture-Detection-Code
#
# Using the Python port of the Matlab Toolbox Complex Shearlet-Based Ridge and Edge Measurement by Rafael Reisenhofer: https://github.com/rgcda/PyCoShREM
#
# This sosftware is distributed under the GNU General Public Licence version 3 (GPLv3) with CSIRO Disclaimer.
# Please see license file for full restrictions.
#
# (c) CSIRO 2022 
#
# Author: Ulrich Kelka

import asdc
await asdc.connect()
asdc.task_select()
from ipywidgets import widgets

project_id, task_id = asdc.get_selection()

import os
import sys
sys.path.append('src')
from processing import *
from tools import *

Tools.GetAssets(Tools, project_id, task_id, )

Tools.SelectAsset(Tools)

Tools.Prepare4Processing(Tools)

hist, proc_hist = PrepareImages(Tools)  
Tools.PlotPreparedImg(Tools, hist, proc_hist)

Tools.SelectEnhancement(Tools)

ReadImage(Tools)
Tools.ShowImage(Tools.DATA2)  

Tools.WriteImage(Tools.DATA2, 'Sobel_test')

Tools.TileImage(Tools)

# # Select System Parameters
# Taken from 'The CoShREM Toolbox Parameter Guide' by Rafael Reisenhofer.\
# http://www.math.uni-bremen.de/cda/software/CoShREM_Parameter_Guide.pdf
#
# ### waveletEffSupp
#
# Length of the effective support in pixels of the Mexican hat wavelet ψ used in the construction the generating shearlet ψgen(x, y) = ψ(x)φ(y), where φ is a Gaussian. The effective support is the interval on which the values of ψ significantly differ from 0. It is, however, not a strictly defined property. A good choice for this parameter is often 1/8 of the image width. If the edges/ridges in the processed image are visible on a large scale, this value should be large relative to the width and height of the processed image.
#
# ### gaussianEffSupp
# Length of the effective support in pixels of the Gaussian φ used in the construction of the generating shearlet ψgen(x, y) = ψ(x)φ(y), where ψ is a Mexican hat wavelet. Typically, this value is chosen to be roughly the half of waveletEffSupp. However, if the edges/ridges in the processed image consist of smooth curves, it can be chosen larger.
#
# ### scalesPerOctave
# Determines the number of intermediate scales for each octave. If scalesPerOctave is set to n, for each orientation, there will be n differently scaled shearlets within one octave.
#
# ### shearLevel (orientations)
# Determines the number of differently oriented shearlets on each scale. If shearLevel is set to n, there will be 2n + 2 differently sheared shearlets on each scale, completing a 180◦ semi-circle.
#
# ### alpha (orientations)
# This parameter can take any value between 0 and 1 and governs the degree of anisotropy introduced via scaling. Roughly speaking, it determines how much the Gaussian is squeezed relative to the wavelet, when scaling the generating shearlet. Formally, the n-th octave is defined by ψn(x, y) = ψgen(2nx, 2αny). For alpha = 0, the degree of anisotropy is maximized while for alpha = 1, both directions are treated the same.
#
# ### octaves
# The number of octaves spanned by the shearlet system. When scales- PerOctave is greater than 1, this parameter can also take non-integer values.

Tools.SystemCombinations(Tools)

GenerateSystems(Tools)                        

# # Detection Parameters
#
# ### minContrast
# Specifies the minimal contrast for an edge/ridge to be detected.
#
# ### offset
# This parameter defines a scaling offset between the even- and odd- symmetric shearlets measured in octaves. If offset = x, the first even-symmetric shearlet used for the computation of the complex shearlet-based edge measure is already x octaves above the first odd- symmetric shearlet considered. In the case of the ridge measure, the converse is true.
#
# ### scalesUsedForPivotSearch
# This parameter defines which scales of the shearlet system are considered for determining the orientation for which the complex shearlet-based edge/ridge measure is computed at a specific location. It can take the values ’all’, ’highest’, ’lowest’ and any subset B ⊂ {1, . . . , scalesPerOctave·octaves}.

Tools.DetectionCombinations(Tools)

DetectFeatures(Tools)

Tools.MosaicTiles(Tools)

Tools.ShowOverlay(Tools)

Tools.SelectFilename(Tools)

Tools.WriteImage(Tools.FEATURES, Tools.FILENAME)

# # Clean feature ensemble
#
# ### min pixel value
# The minimum pixel value in the normalized feature ensemble. 
#
# ### kernel size
# The size of the kernel used for morhological operations (closing, opening, erosion). 
#
# ### min cluster size
# The minimum size of pixel clusters to keep in the image. 

Tools.Enhancement(Tools)

EnhanceEnsemble(Tools)
Tools.ShowCompare(Tools)

Tools.SelectFilename(Tools, 'tif')

Tools.WriteImage(Tools, Tools.FEATURES, Tools.GEOTIF)

# ## Vectorize ensemble map
# We use sknw.py to build a network from the skeleton.
# The pyhton package used is: https://github.com/Image-Py/sknw
#
# It is distributed under BSD 3-Clause License
#
# Copyright (c) 2017, Yan xiaolong

Tools.SelectFilename(Tools, 'shp')

Tools.SHP = BuildSHP(Tools.E_FEATURES, Tools.SHP, 100)


# # Save Assets
# Save assets to WebODM

# +
#Upload example
task_name = asdc.task_dict[task_id]['name']
task_id = asdc.new_task(f"{task_name} - Fracture Detection Output")

#if os.path.isfile(Tools.FILENAME):
#    r = asdc.upload_asset(Tools.FILENAME, dest="odm_orthophoto/odm_orthophoto.tif", task=task_id)
#    print(r)

if os.path.isfile(Tools.GEOTIF):
    r = asdc.upload_asset(Tools.GEOTIF, dest="odm_orthophoto/odm_orthophoto.tif", task=task_id)
    print(r)

if os.path.isfile(Tools.SHP):
    r = asdc.upload_asset(Tools.SHP, dest=f"shapefiles/{Tools.SHP}", task=task_id)
    print(r)



