# -*- coding: utf-8 -*-
"""
GDAL workaround for Polyline_to_Shape.m
Geometry is read from csv file and written to shp
Line geometries are simplified based on the tolerance given

@author: kel321
"""

import sys
import csv
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

def WritePolyline2SHP(inputFile, outputFile, refWKT = None, tolerance=1e-3): 
    path = inputFile

#read in the polylines from csv
    poly_lines = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')     
        for i, row in enumerate(reader): 
            if any(row):  
                if i < 1: 
                    continue 
                s = int(len(row)/2)
                cur_line = []
                for x in range(s):
                    cur_line.append((float(row[x]), float(row[s+x])))
            poly_lines.append(cur_line)
            
#get the GDAL driver for ESRI shapefile
    driverName = "ESRI Shapefile"
    drv = gdal.GetDriverByName( driverName )
    if drv is None:
        print ("%s driver not available.\n" % driverName)
        sys.exit( 1 )
        
#  Create the shp-file (or open it if it exists)
    ds = drv.Create(outputFile, 0, 0, 0, gdal.GDT_Unknown )
    if ds is None:
        print ("Creation of output file failed. Trying to open file...\n")
        ds = drv.Open("point_test.shp", 1) # 0 means read-only. 1 means writeable.
        if ds is None:
            print ("Creation of output file failed.\n")
            sys.exit( 1 )
            
# get layer and check if reference is given (write non-georeferenced shp if not defined)     
    lyr = ds.GetLayer(0)
    if lyr is None:
        if refWKT:
            print("refWKT")
            SpatialRef = osr.SpatialReference()
            SpatialRef.ImportFromWkt(refWKT)
            lyr = ds.CreateLayer( "fitted_polylines", SpatialRef, ogr.wkbLineString)  
        else:
            lyr = ds.CreateLayer( "fitted_polylines", None, ogr.wkbLineString)
        if lyr is None:
            print ("Layer creation failed.\n")
            sys.exit( 1 )    
    lyr.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))      

#loop though the polylines and create lines in the layer
    for i, l in enumerate(poly_lines): 
        feat = ogr.Feature( lyr.GetLayerDefn())
        feat.SetField( "id", i )
        line = ogr.Geometry(ogr.wkbLineString)
        for p in l:
            if refWKT is None:
                line.AddPoint(-p[0], p[1])  #need to flip x coodinates in case of non-georeferenced data
            else:
                line.AddPoint(-p[0], p[1])
        simpleLine = line.Simplify(tolerance)   #simplyfy the line geometry
        simpleLine.SwapXY
        feat.SetGeometry(simpleLine)
        lyr.CreateFeature(feat) 
        feat.Destroy()
    ds = None

if __name__ == '__main__':
    # create WGS84 Spatial Reference
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    refWKT = sr.ExportToWkt()
    print(refWKT)
    
    inputTXT = 'C:\DRONE\Python_API\TEST_1\output\Fitted_Curves\Bingie_Bingie_area2.txt'
    outputFilename = 'output.shp'
   # refWKT = None
    tolerance = 1e-3
    
    WritePolyline2SHP(inputTXT, outputFilename, refWKT, tolerance)