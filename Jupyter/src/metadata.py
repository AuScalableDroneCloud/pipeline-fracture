#
# writes metadata to the DAP (daptst) as a draft to the exsiting collection id: 80194
#

import requests
import sys, os
import json
from datetime import datetime, date

#TODO: How do we get this to work with lists?
#Add flag to write JSON


def datetime_to_isoformat(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()



def CreateMetaData(Tools):
    if (Tools.JSON == True and Tools.DAP == False):
        print('writing JSON')
        CreateJSON(Tools)
    if (Tools.DAP == True):
        print('writing JSON and publishing to CSIRo DAP (test).')
        WriteCollection(Tools)
    if (Tools.TERN == True):
        print('publishing to TERN storage not implemented')

def CreateJSON(Tools):
    if (len(Tools.FILE) > 0):
        name = str(Tools.J_NAME) + '.json'
        in_name   = Tools.FILE[0]
        out_name1 = Tools.GEOTIF
        out_name2 = Tools.SHP
    
        m_data = {
        	"name": "Structural Geology",
        	"description": "Fracture detection with Complex Shearlet Transform based on https://github.com/rahulprabhakaran/Automatic-Fracture-Detection-Code Using the Python port of the Matlab Toolbox Complex Shearlet-Based Ridge and Edge Measurement by Rafael Reisenhofer: https://github.com/rgcda/PyCoShREM",
        	"notebook": {
        		"file": "CoSh_ensemble_webodm.ipynb",
        		"version": 1.0,
        		"parameters": {
                    "Projection ": Tools.PROJ,
                    "Geotransform ": Tools.GEOT,
                    "Size ": Tools.EXTEND,
                    "edges": Tools.EDGES, 
                    "ridges": Tools.RIDGES,
                    " positive": Tools.POSITV,
                    " negative": Tools.NEGATI,
        			"waveletEffSupp": str(Tools.WAVEEF)  + " px",
        			"gaussianEffSupp": str(Tools.GAUSEF) + " px",
        			"scalesPerOctave": Tools.SCALES,
        			"shearLevel": Tools.SHEARL,
        			"alpha": Tools.ALPHA,
        			"octaves": Tools.OCTAVE,
    
        			"minContrast": Tools.MINCON,
        			"offset": Tools.OFFSET,
        			"scalesUsedForPivotSearch": Tools.PIVOTS,
                    
        			"min pixel value": Tools.THRESH,
        			"min cluster size": str(Tools. MINSI) + " px"
        		},
        		"assets": [{
            		"type": "input",
                    "title": "Structural Geology",
                    "creator": "Uli Kelka",
            		"description": "orthomosaic",
            		"name": in_name,
            		"format": str(os.path.splitext(in_name)[1])
        			},     
        			{
        				"type": "output",
                        "title": "Structural Geology",
                        "creator": "Uli Kelka",
        				"description": "Intensity map",
        				"name": out_name1,
        				"format": str(os.path.splitext(out_name1)[1])
        			},  
                    {
        				"type": "output",
                        "title": "Structural Geology",
                        "creator": "Uli Kelka",
        				"description": "shape file",
        				"name": out_name2,
        				"format": os.path.splitext(out_name2)[1]    #unnecessary as this is always a shp :-)
        			}
        		]
        	},
        	"author": "Uli Kelka",
        	"organisation": "CSIRO ",
        	"licence": {
        		"name": "It is distributed under BSD 3-Clause License"
        	},
        	"run": {
        		"date": datetime_to_isoformat (date.today() )
        	}
        }
        with open(name, "w") as outfile:
            json.dump(m_data, outfile)
    else:
        print('No input file defined!')
        sys.exit()
    return(m_data)
    
def WriteCollection(Tools):
    collectionId = "80194"
    #REST credentials
    username = Tools.USER
    password = Tools.PASSW    
    auth = requests.auth.HTTPBasicAuth( username, password )
    headers_object = {"Accept":"application/json"}
    
    # get collection metadata
    url = "https://daptst.csiro.au/dap/api/v2/collections/"+collectionId
    r = requests.get(url, auth=auth)
    if r.ok:
        print("Collection "+collectionId+" metadata accessed")
    else:
        print("Something went wrong!")
    metadata = r.json()
    new_metadata = metadata.copy()  #Copy the metadata dict.
    #print(json.dumps(new_metadata, indent=2))
    
    #Create dummy metadata
    metadata = CreateJSON(Tools)
    new_metadata["description"] = metadata.get("description")
    new_metadata["credit"] = metadata.get("author")
    notebook = metadata.get("notebook")
    new_metadata["lineage"] = "notebook:"+notebook.get("file")+"\nversion:"+str(notebook.get("version"))+"\nparameters: "+str(notebook.get("parameters"))
    #print(json.dumps(new_metadata, indent=2))
    save_request = requests.put(url,
        auth=auth,
        headers=headers_object,
        json=new_metadata)
    #print("Response code: {0}".format(save_request.status_code))
    
    #print(metadata.get("name"))
    notebook = metadata.get("notebook")
    assets = notebook.get("assets")
    index = 0
    files = []
    old_type = ""
    print("Uploading "+str(len(assets))+" assets...")
    writeFiles = False
    for asset in assets:
       index = index +  1
       #print("*** asset '{0}' '{1}'".format(index,asset))
       filename = asset.get("name")
       type = asset.get("type")
       print(str(index)+": "+type+" - "+filename)
       if (type == old_type) or (old_type == ""):
          #print(str(index)+": "+type+" - "+filename)
          files.append( ('file',(os.path.split(filename)[1],open(filename,'rb')) ) )
       else:
          url = "https://daptst.csiro.au/dap/api/v2/collections/"+collectionId+"/files?path=/"+old_type
          writeFiles = True
    
       if (index == len(assets)) and not(writeFiles):
          url = "https://daptst.csiro.au/dap/api/v2/collections/"+collectionId+"/files?path=/"+type
          writeFiles = True
       if writeFiles: 
          r = requests.post(url, auth=auth, files=files)
          if not r.ok:
             print("FILES: Something went wrong!")
             print(r.text)
          files = []
          files.append( ('file',(os.path.split(filename)[1],open(filename,'rb')) ) )
          writeFiles = False
       old_type = type
    
    # add metadata for each of the files(assets)  uploaded previously
    index = 0
    for asset in assets:
       filename = asset.get("name")
       type = asset.get("type")
       collectionFolder = "/"+type
       url = "https://daptst.csiro.au/dap/api/v2/collections/"+collectionId+"/file?path="+collectionFolder+"/"+os.path.split(filename)[1]
       print("Adding metadata to asset:"+collectionFolder+"/"+os.path.split(filename)[1])
       r = requests.get(url, auth=auth)
       if not r.ok:
           print("Something went wrong!")
    
       metadata = r.json()
       fileId = metadata.get("id")
       if not fileId:
           print("ERROR: POST request to '{0}' ".format(url) \
               + "did not contain a fileId in the response.")
           #You would need some error handling here.
       #print("fileId: {0}".format(fileId))
    
       params = metadata.get("parameters")
       #print(json.dumps(params, indent=2))
    
       index=0
       # if no params (typically if not an image file), need to add at least title and creator fields
       if (len(params) == 0):
          params.append( { "name": "Title", "dateValue": "", "dateValueString": "", "numericValue": "", "stringValue": "" } )
          params.append( { "name": "Creator", "dateValue": "", "dateValueString": "", "numericValue": "", "stringValue": "" } )
          params.append( { "name": "Description", "dateValue": "", "dateValueString": "", "numericValue": "", "stringValue": "" } )
       for param in params:
         #print("*** param '{0}' '{1}'".format(index,param))
         if param.get("name") == "Title":
           param["stringValue"] =  asset.get("title")
           params[index] = param
         if param.get("name") == "Creator":
           param["stringValue"] = asset.get("creator")
           params[index] = param
         if param.get("name") == "Creation Date":
           param["stringValue"] = datetime.now().strftime("%Y-%m-%d")
           params[index] = param
         if param.get("name") == "Description":
           param["stringValue"] = asset.get("description")
           params[index] = param
         if param.get("name") == "Format":
           param["stringValue"] = asset.get("format")
           params[index] = param
         if param.get("name") == "Coverage":
           param["stringValue"] = "coverage"
           params[index] = param
         if param.get("name") == "Source":
           param["stringValue"] = "source"
           params[index] = param
         if param.get("name") == "Subject":
           param["stringValue"] = "rock fracture"
           params[index] = param
         if param.get("name") == "Identifier":
           param["stringValue"] = "identifier"
           params[index] = param
         index = index + 1
    
       metadata["parameters"] = params
       
       url = "https://daptst.csiro.au/dap/api/v2/collections/"+collectionId+"/files/{0}".format(fileId)
    
       save_request = requests.put(url,
           auth=auth,
           headers=headers_object,
           json=metadata)
     
