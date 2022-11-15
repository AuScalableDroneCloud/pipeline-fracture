import asdc
await asdc.connect()
asdc.task_select()
#asdc.project_select()
from ipywidgets import widgets

project_id, task_id = asdc.get_selection()

# +
#Upload example
task_id = asdc.new_task("Example orthophoto")

#Just upload the example to test this (converted to .tif)
from PIL import Image
image = Image.open("test/Ortho_3_061.png")
image.save("test/orthophoto.tif")
#r = asdc.upload_asset("test/orthophoto.tif", dest="odm_orthophoto/", task=task_id) #This would use orthophoto.tif as dest filename, so provide below
r = asdc.upload_asset("test/orthophoto.tif", dest="odm_orthophoto/odm_orthophoto.tif", task=task_id)
print(r)
# -

print(task_id)

#Attempt to download
fn = asdc.download_asset('orthophoto.tif', dest=f"orthophoto_{task_id}.tif")
#fn = asdc.download_asset('odm_orthophoto/orthophoto.tif', dest=f"orthophoto_{task_id}.tif")
print(fn)


