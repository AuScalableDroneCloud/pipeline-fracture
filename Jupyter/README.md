The software requires [gdal](https://pypi.org/project/GDAL/), [openCV](https://pypi.org/project/opencv-python/), and [PyCoShREM](https://github.com/rgcda/PyCoShREM).

# ASDC - CoSheRem
Australian Scalable Drone Cloud - Complex Shearlet Ridge and edge measure. This software is distributed under the GPLv3 license with CSIRO disclaimer. The purpose of this software is to automatise the feature detection in any image and is particularly designed for the detection of geological discontinuities. In UAV derived orthomosaics this could be fractures but the workflow is also applicable to surface models or digital elevation models. The execution can be performed either via the python script (ASDC_CoSheRem.py) or in a more interactive manner via the Jupyter notebook (ASDC_CoSheRem.ipynb). A cloud-hosted version is available [here](https://asdc.cloud.edu.au) as a processing pipeline for UAV derived images. 
The workflow is part of the [ASDC](https://asdc.io/).
![image](https://user-images.githubusercontent.com/82503083/190591385-dca6b50e-7555-4ebe-9c2c-99d71b9dbc34.png)

Below are the different processing steps are outlined. In bold are the names of the parameters as they appear in the Jupyter notebook, the keywords in brackets at the end are the optional commands that can be passed via command line when executing the python script. Please note that image tiling is only supported via (gdal_retile.py) for single images in the notebook. If image lists are passed to the python file (--file), no white spaces are allowed, and the names need to be comma-separated.

## ASDC_CoSheRem.py specific
**--file** filename or list of comma-separated names without white spaces  <br />
**--out1** filename of the feature intensity map  <br />
**--out2** filename of the shp file file <br />

## Prepare Image
Reads in the image and converts it to a single channel grayscale.
Currently up to four channel images are supported with the option below:

**Resize:** *Resized the image based on the value given here in percent* (--resize) <br />
**DetailEnhancement:** *Performs detail enhancement based on the sigma r and sigma s values given.* (--sigR; --sigS) <br />
**GammaCorrection:** *Factor to perform gamma correction with.* (--gamma) <br />
**WhiteBalance:** *White balance the image by applying Gray world white balance algorithm* (--wb) <br />

## Enhance Image
Enhance the image prior to feature detection. 
The currently implemented options are: <br/>
**Histogram equalization:** *Performs a histogram equalization.* (--histEq) <br />
**Gaussian blur:** *Gaussian blur using a 5x5 kernel.* (--gaussB) <br />
**Sharpen:** *Sharpen the image using a 3x3 kernel.* (--sharpen) <br />
**Sobel:** *Calculates the mean the mean horizontal and vertical gradient of the image using 5x5 kernels respectively.* (--sobel) <br />
**Edge:** *Perform a simple edge detection using a 5x5 kernel.* * (--edge) <br />
 **Invert:** *Inverts the image using bit-wise inversion.* (--invert) <br />
 
## Generate systems
Building shearlet systems for the images(s) based on the parameter below. Note that lists and every possible parameter combination will be generated. This allows muti-scale edge/ridge detection to be performed on the input images as suggested by [Prabhakaran etla., 2019](https://doi.org/10.5194/se-10-2137-2019) <br />
For more detailed information about the parameters click [here](http://www.math.uni-bremen.de/cda/software/CoShREM_Parameter_Guide.pdf). <br />
**waveletEffSupp:** *Define the pixel length of Mexican hat wavelets used for constructing the systems.* (--wave) <br />
**gaussianEffSupp:** *Pixel length of the Gaussian used in the construction of the shearlet.* (--gaus) <br />
**scalesPerOctave:** *Number of intermediate scales for each octave.* (--scal) <br />
**shearLevel:** *Number of differently oriented shearlets at each scale.* (--shea) <br />
**alpha:** *Parameter governing the degree of anisotropy introduced via scaling.* (--alph) <br />
**octaves:** *Number of octaves spanning the shearlet system.* (--octa) <br />
**Ridges:** *Detect ridges in the image* (--ridges) <br />
**Edges:** *Detect edges in the image* (--edges) <br />

## Detect features
Features are detected in the images with the generated shaerlet systems. Each generated systems will be used to generate a ridge/edge intensity map that is then normalized. The parameters are: <br />
**minContrast:** *Minimum contrast of edges/ridges to be detected* (--minC) <br/>
**offset:** *Defines the scaling offset between even- and odd-symmetric shearlets.* (--offS) <br />
**scalesUsedForPivotSearch:** * *Defines which scales of the shearlet systems are considered for determining the orientation for which the complex shearlet-based edge/ridge measure is computed.* (This parameter can only be changed in the Jupyter notebook) * (This parameter can only be changed in the Jupyter notebook) <br />
**positive:** *Detect positive ridges.* (--positive) <br />
**negative:** *Detect negative ridges.* (--negative) <br />

## Filter edges/ridges
The feature intensity map can be filtered based on a pixel value threshold and considering the size of connected pixel clusters. The two parameters are: <br />
**min pixel value:** *Features below this threshold are omitted.* (--thresh) <br />
**min cluster size:** *The minimum size in pixels of clusters to keep in the image.* * (--minS) <br />
The resulting feature collection will form the input for the **skeletonization**. 

## Polylinefitting
The fitting of polylines to teh skeletonized feature collection is performed using [skn.py](https://github.com/Image-Py/sknw) by Yan xiaolong.
The software is distributed under a BSD 3-Clause License.

## Example
Example images of fracture networks obtained from <br />
https://research.tudelft.nl/en/datasets/fracture-network-patterns-from-the-parmelan-anticline-france-2 and <br />
https://publications.rwth-aachen.de/record/793416?ln=en 
![image](https://user-images.githubusercontent.com/82503083/190591211-6fe74e9f-1570-4ebc-8837-c9f71c08f9a2.png)
