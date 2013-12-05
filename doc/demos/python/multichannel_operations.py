from smilPython import *

# Load a RGB image
im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/arearea.png")
im1.show()

# Copy the green channel into a UINT8 image
im2 = Image()
copyChannel(im1, 1, im2)
im2.show()

# Split RGB channels into a 3D UINT8 image with 3 slices (one for each channel)
im3 = Image()
splitChannels(im1, im3)
im3.show()

# Perform a 2D dilation on the slices
im4 = Image(im3)
dilate(im3, im4)
im4.show()

# And merge the result into a RGB image
im5 = Image(im1)
mergeChannels(im4, im5)
im5.show()
