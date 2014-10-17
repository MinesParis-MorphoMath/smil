#modules.
import sys
sys.path.append ("/usr/local/lib/Smil")
from smilPython import *
import bpy

#Create Topology from 2D-Pictures.
smil_image_index=None
def topography (image):
    global smil_image_index
    if smil_image_index is None:
        smill_image_index = 0
    name = image.getName()
    if name == '':
        name = 'smil_image_' + str(smil_image_index)
        index += 1
    bpy.data.meshes.new (name)
    
    nbr_pixel = image.getPixelCount ()
    count = 0
    for i in range(nbr_pixel):
        count += 1


#create Arrows
#def arrows (imageGray, imageArrow):

#Create Plane from pixels.
#def plane (list_pixels): 


topography (Image ("http://cmm.ensmp.fr/~faessel/smil/images/lena.png"))
