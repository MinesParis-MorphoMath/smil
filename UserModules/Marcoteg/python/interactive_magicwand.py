from smilPython import *
#from tp_init import *
from marcoteg_utilities_smil import *
from smilMarcotegPython import *
import pdb
import math
def CoordsFromOffset(im,offset):
    ypos = int(math.floor(offset/im.getWidth()))
    xpos = int(offset - ypos * im.getWidth())
    return xpos,ypos
# --------------------------------------------------
# --------------------------------------------------
# SimpleMagicWand : takes the pixel value val(x,y), applies a
# threshold [val-tolerance, val+tolerance] and selects the CC of
# pixel(x,y)
# --------------------------------------------------
# --------------------------------------------------
def SimpleMagicWand(imin,x,y,tolerance,imOut,nl=Morpho.getDefaultSE()):
    imThresh =Image(imin)
    imPoint =Image(imin)
    val = imin.getPixel(x,y)
    imPoint<<0
    imPoint.setPixel(x,y,255)
    thresh = val+ tolerance
    compare(imin,"<",thresh, 255,0,imThresh)
    
    if(val > tolerance):
        thresh = val- tolerance
    else:
        thresh = 0
    compare(imin,">",thresh, 255,0,imOut)	
    inf(imThresh,imOut,imThresh)
    labelWithMeasure(imThresh,imPoint,imOut,"max",nl)
#    labelWithMax(imThresh,imPoint,nl,imOut)


def magicWandSegInit(im,nl=Morpho.getDefaultSE()):
    imgra,imFineSeg = Image(im),Image(im,"UINT16")
    gradient(im,imgra,nl)
    g = watershedExtinctionGraph(imgra,imFineSeg,"v",nl)
    return g,imFineSeg
    
def magicWandSegUse(im,g,imFineSeg,xpos,ypos,tolerance,imMW,nl=Morpho.getDefaultSE()):

    imMW16=Image(im,"UINT16")
    imSeg,imRes16=Image(imMW16),Image(imMW16)

    # Get CC with homogeneous gray level (tolerance) around pixel (xpos,ypos)
    SimpleMagicWand(im,xpos,ypos,tolerance,imMW,nl)

    #after removing edges, they can not be reactivated.
    # That is why a temporary graph is used
    g_tmp = g.clone() 

    # extend the classic magic wand result to the adaptive hierarchical level
    copy(imMW,imMW16)
    magicWandSeg(imMW16,imFineSeg,g_tmp)

    # get back, from graph to image
    graphToMosaic(imFineSeg, g_tmp, imSeg)
    # if graph functions require 16 bits images, come back to 8 bits resulting image
    labelWithMeasure(imSeg,imMW16,imRes16,"max",nl)
    copy(imRes16,imMW)


if(0):
    im = Image(images_dir+"/Gray/arearea.png")
    imMW = Image(im)
    g,imFineSeg =  magicWandSegInit(im,nl)
    magicWandSegUse(im,g,imFineSeg,xpos,ypos,tolerance,imSeg,nl)


# read input image
im = Image(images_dir+"Gray/tools.png")
imMW = Image(im)
imOverl = Image(im)
imRes = Image(im)
imRes << 0
# parameters
nl = HexSE()
tolerance = 10
xpos,ypos = 154,37


# compute hierarchy for further use
g,imFineSeg =  magicWandSegInit(im,nl)

#    magicWandSegUse(im,g,imFineSeg,xpos,ypos,tolerance,imMW,nl)
im.show()
imRes.show()

v = im.getViewer()

class slot(EventSlot):
    def run(self, event=None):
      v.getOverlay(imOverl)
      listOffsets=nonZeroOffsets(imOverl)
#      watershed(im2, imOverl, im3, im4)
      print "ok"
      i = 0
      prevListOffsets = []
      for offset in listOffsets:
          i = i + 1
          print i, offset
          if offset in prevListOffsets:
              print "continue"
              continue
          xpos,ypos =  CoordsFromOffset(im,offset)
          print xpos,ypos
          magicWandSegUse(im,g,imFineSeg,xpos,ypos,tolerance,imMW,nl)

          compare(imMW,">",0,imMW,imRes,imRes)
      print "--------------"
      prevListOffsets = listOffsets        
s = slot()

v.onOverlayModified.connect(s)
v.onOverlayModified.trigger()

print "1) Right click on im"
print "2) In the \"Tools\" menu select \"Draw\""
print "3) Draw markers (with different colors) on im1 and view the resulting segmentation"

# Will crash if not in a "real" Qt loop
Gui.execLoop()
