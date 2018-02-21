from smilPython import *

# Load an image
im1 = Image("http://smil.cmm.mines-paristech.fr/images/balls.png")
im1.show()

# Create the skeleton using a thinning with a combination of the 4 rotations of the composite SE sL1 and sL2
im2 = Image(im1)
fullThin(im1, HMT_sL1(4) | HMT_sL2(4), im2)

# Detect line junctions
im3 = Image(im1)
hitOrMiss(im2, HMT_sLineJunc(8), im3)

# Detect line ends
im4 = Image(im1)
hitOrMiss(im2, HMT_sLineEnd(8), im4)

# Modigy results for display...
dilate(im3, im3, hSE())
dilate(im4, im4, hSE())
inf(im3, 1, im3)
inf(im4, 3, im4)
sup(im3, im4, im4)

im2.getViewer().drawOverlay(im4)
im2.show()

