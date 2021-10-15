#!/usr/bin/env python
"""
Sample script that uses the FractureDetection module created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

from __future__ import print_function
import FractureDetection
import matlab

my_FractureDetection = FractureDetection.initialize()

folderIn = "C:\\DRONE\\Python_API\\TEST_1\\output\\"
rowIn = matlab.double([1000.0], size=(1, 1))
colIn = matlab.double([1000.0], size=(1, 1))
my_FractureDetection.main(folderIn, rowIn, colIn, nargout=0)

my_FractureDetection.terminate()
