from smilPython import *

# Load an image
im1 = Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
im1.show()

# Manual threshold (between 100 and 255)
im2 = Image(im1)
threshold(im1, 100, 255, im2)
im2.show()

# Otsu automatic threshold
im3 = Image(im1)
# Generate two threshold values (i.e. 3 classes)
otsuThreshold(im1, im3, 2)
# Display the resulting image with three labels values corresponding to the three classes
im3.showLabel()
