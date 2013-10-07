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
for bb in bboxes.values():
    drawRectangle(imRec, bb[0], bb[2], bb[1]-bb[0]+1, bb[3]-bb[2]+1, 1)
imIn.getViewer().drawOverlay(imRec)

# Blobs measures
blobs = computeBlobs(imLbl)
# areas
areas = measAreas(imLbl, blobs) # equivalent but faster than measAreas(imLbl)
# barycenters
barys = measBarycenters(imLbl, blobs)
# volume of blobs in imIn
vols  = measVolumes(imIn, blobs)
print "Label\tarea\tvolume\tbarycenter (x,y)"
for lbl in blobs.keys():
  print str(lbl) + "\t" + str(areas[lbl]) + "\t" + str(vols[lbl]) + "\t" + str(barys[lbl])

