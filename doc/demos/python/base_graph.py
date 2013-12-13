from smilPython import *

im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/mosaic.png")
im2 = Image(im1)
im3 = Image(im1)
imMos = Image(im1, "UINT16")
imArea = Image(imMos)
imSeg = Image(imMos)

label(im1, imMos)
labelWithArea(im1, imArea)

g = mosaicToGraph(imMos, imArea)

drawGraph(imMos, g, imSeg)
imMos.getViewer().drawOverlay(imSeg)

g.removeNodeEdges(3)
graphToMosaic(imMos, g, imSeg)

im1.show()
imMos.showLabel()
imSeg.showLabel()
