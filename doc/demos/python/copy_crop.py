from smilPython import *

im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/barbara.png")
im2 = Image("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
im3 = Image(im2)

im1.show()
im2.show()
im3.show()

# Copy the content of im1 from (256, 0) in im3
copy(im1, 256, 0, im3)

# Copy the content of im2 and put it at position (0, 256) in im1
copy(im2, im1, 0, 256)

# Copy the window starting at (256, 0) and with dimensions 128x128 and put it at (128, 128) in im2
copy(im1, 256, 0, 128, 128, im2, 128, 128)
# Same as previous (simple way)
copy(im1, 256, 0, im2, 128, 128)

# Create a 3D image and copy slices inside
im3D = Image(im1.getWidth(), im1.getHeight(), 4)
im3D << 0
copy(im1, im3D)
copy(im2, im3D, 128, 128, 1)
copy(im3, im3D, 128, 128, 2)
im3D.show()