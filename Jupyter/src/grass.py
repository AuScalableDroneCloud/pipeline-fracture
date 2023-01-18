# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:45:39 2022

@author: kel321
"""

import grass.script as grass




def GrassNoData():
    
    grass.run_command('grass:r.mapcalculator',
                      {"amap": gPb_rlayer,
                       "formula": "if(A>0, 1, null())",
                       "GRASS_REGION_PARAMETER": "%f,%f,%f,%f" % (xmin, xmax, ymin, ymax),
                       "GRASS_REGION_CELLSIZE_PARAMETER": 1,
                       "outfile": mapcalc})
    pass

def GrassThinning():
    pass

def GrassR2Vec():
    pass







processing.runalg('grass7:r.thin',
                  {"input": mapcalc,
                   "GRASS_REGION_PARAMETER": "%f,%f,%f,%f" % (xmin, xmax, ymin, ymax),
                   "output": thinned})

processing.runalg('grass7:r.to.vect',
                  {"input": thinned,
                   "type": 0,
                   "GRASS_OUTPUT_TYPE_PARAMETER": 2,
                   "GRASS_REGION_PARAMETER": "%f,%f,%f,%f" % (xmin, xmax, ymin, ymax),
                   "output": centerlines})