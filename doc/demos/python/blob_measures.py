from smilPython import *

# Load an image
imIn = Image("http://cmm.ensmp.fr/~faessel/smil/images/DNA_small.png")
imIn.show()

imThr = Image(imIn)
threshold(imIn, imThr)

imLbl = Image(imIn, "UINT16")
label(imThr, imLbl)
imLbl.showLabel()

# Bounding boxes
bboxes = measBoundBoxes(imLbl)
imRec = Image(imIn)
drawRectangles(imRec, bboxes)
imIn.getViewer().drawOverlay(imRec)

# Blobs measures
blobs = computeBlobs(imLbl)
# areas
areas = measAreas(imLbl, blobs) # equivalent but faster than measAreas(imLbl)
# barycenters
barys = measBarycenters(imLbl, blobs)
# volume of blobs in imIn
vols  = measVolumes(imIn, blobs)
print("Label\tarea\tvolume\tbarycenter (x,y)")
for lbl in blobs.keys():
  print(str(lbl) + "\t" + str(areas[lbl]) + "\t" + str(vols[lbl]) + "\t" + str(barys[lbl]))

