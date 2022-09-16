# ASDC - CoSheRem
Australian Scalable Drone Cloud - Complex Shearlet Ridge and edge measure

![pic](wf.png)

## Prepare Image
Reads in the image and converts it to a single channel gray scale.
Currently up to four channel images are supported with teh option below:

Resize: *Resized the image based on teh value given pere in percent* (--resize) <br />
DetailEnhancement: *Perfomes detail enhancemnt baed on the sigma r and sigma s values given* (--sigR; --sigS) <br />
GammaCorrection: *Factot to perfom gamma correction with* (--gamma) <br />
WhiteBalance: *White balance image by applying Gray world while balance algorithm* (--wb) <br />

## Enhance Image
Enahnce the image prior to feature detection. The currently implemented options are:

Histogram equalization : *Perfoms a histogram equalization* (--histEq) <br />
Gaussian blur: *Perfomes gaussina blur on the image using a 5x5 kernel.* (--gaussB) <br />
Sharpen: *Performs a 2D filtering of the image using a 3x3 kernel. (--sharpen) <br />
-1 -1 -1 <br />
-1  9 -1 <br />
-1 -1 -1 <br />
Sobel: *Calulates the mean the mean horizontal and vertical gradient of the image using 5x5 kernels respectively.* (--sobel) <br \>
Edge: *Perfomes a 2D filtering operation on the image to perfom a simple edge detection using the 5x5 kernel below.* (--edge) <br \>
 0  0 -1  0  0 <br \>
 0 -1 -2 -1  0 <br \>
-1 -2 16 -2 -1 <br \>
 0 -1 -2 -1  0 <br \>
 0  0 -1  0  0 <br \>
 Invert: *Inverts the image using bit-wise inversion.* (--invert) <br />
 
## Generate systems
Building shearlet systems for the images(s) absed ont eh parameter below. Note that lists and every possible parameter combination will be generated. Thsi allows muti-scale edge/ridge detection to be perfomed on the input images. For more detailed information ont eh parameters click [here](http://www.math.uni-bremen.de/cda/software/CoShREM_Parameter_Guide.pdf). <br \>
waveletEffSupp: *Define the pixel length of Mexican hat wavelets used fro constructing teh systems.* (--wave) <br />

 
