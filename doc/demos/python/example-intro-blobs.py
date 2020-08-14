import smilPython as sp

im = sp.Image("balls.png")
iml = sp.Image(im)

# this is just a binary image - no need to segment
sp.label(im, iml)
iml.showLabel()

# create blobs structure
blobs = sp.computeBlobs(iml)

# evaluate some measures :
areas = sp.measAreas(blobs)
barys = sp.measBarycenters(im, blobs)

# print areas and barycenters of each region
print("{:3s} - {:>6s} - {:>13s}".format("ID", "Area", "Barycenter"))
for k in blobs.keys():
  bary = barys[k]
  print("{:3d} - {:6.0f} - {:6.0f} {:6.0f}".format(k, areas[k], bary[0], bary[1]))

