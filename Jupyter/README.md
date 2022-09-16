# ASDC - CoSheRem
Australian Scalable Drone Cloud - Complex Shearlet Ridge and edge measure.
This software is distributed under the GPLv3 license with CSIRO disclaimer. 
The purpose of this software is to automatise the feature detection in any image and is particularilly designed for teh detection of geological discontinuities. In UAV deved orthomosaics this could be fracures but the workflow is alos applicabe to surface models or digital elevation models.
The excetution can be performed eiterh via the pyhton script (ASDC_CoSheRem.py) or in a more interactive manner via the Jupyter notebook (ASDC_CoSheRem.ipynb). A cloud-hosted version is availabe [here](https://asdc.cloud.edu.au) as a processing pipleine fro UAV derived images.
The workflow is part of [ASDC](https://asdc.io/).
![image](https://user-images.githubusercontent.com/82503083/190591385-dca6b50e-7555-4ebe-9c2c-99d71b9dbc34.png)
## Prepare Image
Reads in the image and converts it to a single channel gray scale.
Currently up to four channel images are supported with the option below:

**Resize:** * Resized the image based on the value given here in percent* (--resize) <br />
**DetailEnhancement:** * Perfomes detail enhancemnt based on the sigma r and sigma s values given.* (--sigR; --sigS) <br />
**GammaCorrection:** * Factor to perform gamma correction with.* (--gamma) <br />
**WhiteBalance:** * White balance the image by applying Gray world white balance algorithm* (--wb) <br />

## Enhance Image
Enhance the image prior to feature detection. 
The currently implemented options are: <br/>
**Histogram equalization:** * Perfoms a histogram equalization* (--histEq) <br />
**Gaussian blur:** * Perfomes gaussina blur on the image using a 5x5 kernel.* (--gaussB) <br />
**Sharpen:** * Performs a 2D filtering of the image using a 3x3 kernel. (--sharpen) <br />
**Sobel:** * Calculates the mean the mean horizontal and vertical gradient of the image using 5x5 kernels respectively.* (--sobel) <br />
**Edge:** * Perfomes a 2D filtering operation on the image to perfom a simple edge detection using the 5x5 kernel.* (--edge) <br />
 **Invert:** * Inverts the image using bit-wise inversion.* (--invert) <br />
 
## Generate systems
Building shearlet systems for the images(s) absed ont eh parameter below. Note that lists and every possible parameter combination will be generated. Thsi allows muti-scale edge/ridge detection to be perfomed on the input images as suggested by [Prabhakaran etla., 2019](https://doi.org/10.5194/se-10-2137-2019) <br />
For more detailed information about the parameters click [here](http://www.math.uni-bremen.de/cda/software/CoShREM_Parameter_Guide.pdf). <br />
**waveletEffSupp:** * Define the pixel length of Mexican hat wavelets used fro constructing teh systems.* (--wave) <br />
**gaussianEffSupp:** * Pixel length of the Guassian used in the construction of the shearlet.* (--gaus) <br />
**scalesPerOctave:** * Number of intermediate scales for each octave.* (--scal) <br />
**shearLevel:** * Number of differently oriented shearlets at each scale.* (--shea) <br />
**alpha:** * Parameter governing the degree of anisotropy intriduced via scaling* (--alph) <br />
**octaves:** * Number of octaves spanning the shearlet system.* (--octa) <br />
**Ridges:** * Detect ridges in the image* (--ridges) <br />
**Edges:** * Detect edges in the image* (--edges) <br />

## Detect features
Features are detected in the images with the generated shaerlet systems. Each generated systems will be used to generate a ridge/edge intesity map that is then normalized. The parameters that can be chosed are: <br />
**minContrast:** * Minimum contrast of edges/ridges to eb detected* (--minC) <br/>
**offset:** * Defines teh scaling offset between even- and odd-symmetric shearlets.* (--offS) <br />
**scalesUsedForPivotSearch:** * Defines which scales of the shearlet systems are considered for determining the orientationn for which the complex shearlet-based edge/ridge measure is computed.* (This parameter can only be changed in the Jupyter notebook) <br />
**positive:** * Detect positive ridges.* (--positive) <br />
**negative:** * Detect negative ridges.* (--negative) <br />

## Filter edges/ridges
The feature intesity map can be filtered based on a pixel value threshold and considering teh size of connected pixel clusters. The two parameters are: <br />
**min pixel value:** * Features below this threshold are omitted.* (--thresh) <br />
**min cluster size:** * The minimum size in pixels of clsuters to keep in th eimage.* (--minS) <br />
The resutling feature collection will form the input for the **skeletonization**. 

## Polylinefitting
The fitting of polylines to teh skeletonized feature collection is performed using [skn.py](https://github.com/Image-Py/sknw) by Yan xiaolong.
The software is distributed underBSD 3-Clause License.

## Example
Example images of fracture networks obtained from https://research.tudelft.nl/en/datasets/fracture-network-patterns-from-the-parmelan-anticline-france-2
and https://publications.rwth-aachen.de/record/793416?ln=en 
![image](https://user-images.githubusercontent.com/82503083/190591211-6fe74e9f-1570-4ebc-8837-c9f71c08f9a2.png)
