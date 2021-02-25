from smilPython import *

im1 = Image("https://smil.cmm.minesparis.psl.eu/images/barbara.png")
im2 = Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
im3 = Image(im1)

im1.show()
im2.show()
im3.show()

# Crop the content of im1 from (256, 0) to im3 (which will be resized)
crop(im1, 256, 0, 256, 256, im3)

# Copy the content of im2 and put it at position (0, 256) in im1
copy(im2, im1, 0, 256)

# Copy the window starting at (256, 0) and with dimensions 128x128 and put it at (128, 128) in im2
copy(im1, 256, 0, 128, 128, im2, 128, 128)
# Same as previous (simple way)
copy(im1, 256, 0, im2, 128, 128)

# Create a 3D image and copy slices inside
im3D = Image(im2.getWidth(), im2.getHeight(), 3)
im3D << 0
copy(im1, 0, 256, im3D)
copy(im3, im3D, 0, 0, 2)
close(im3D, im3D, cbSE())
im3D.show()
