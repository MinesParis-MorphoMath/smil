from smilPython import *

# Load an image
imIn = Image("http://smil.cmm.mines-paristech.fr/images/DNA_small.png")
imIn.show()

imThr = Image(imIn)
threshold(imIn, imThr)

imLbl = Image(imIn, "UINT16")
label(imThr, imLbl)
imLbl.showLabel()

# Bounding boxes
bboxes = blobsBoundBox(imLbl)
imRec = Image(imIn)
drawRectangles(imRec, bboxes)
imIn.getViewer().drawOverlay(imRec)

# Blobs measures
blobs = computeBlobs(imLbl)
# areas
areas = blobsArea(imLbl, blobs) # equivalent but faster than measAreas(imLbl)
# barycenters
barys = blobsBarycenter(imLbl, blobs)
# volume of blobs in imIn
vols  = blobsVolume(imIn, blobs)
print("Label\tarea\tvolume\tbarycenter (x,y)")
for lbl in blobs.keys():
  print(str(lbl) + "\t" + str(areas[lbl]) + "\t" + str(vols[lbl]) + "\t" + str(barys[lbl]))

