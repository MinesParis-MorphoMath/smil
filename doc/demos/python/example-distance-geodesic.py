import smilPython as sp

im = sp.Image("cells.png")

imDist = sp.Image(im)
imMask = sp.Image(im)
imMark = sp.Image(im)
imGeoDist = sp.Image(im)

# create a marker image, the same as the original image except at
# some point inside the "true" region, which is set to "0" 
nl = sp.HexSE()
sp.distance(im, imDist, nl)
sp.compare(imDist, "==", sp.maxVal(imDist), 0, im, imMark)

# use the original image as the mask.
sp.copy(im, imMask)
sp.distanceGeodesic(imMark, imMask, imGeoDist, nl)

imGeoDist.show()
sp.maxVal(imGeoDist)
