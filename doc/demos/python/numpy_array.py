from smilPython import *
import numpy as np

# Create an image
im1 = Image(256, 256)
im1.show()

# Create a numpy array containing the real image pixels
imArr = im1.getNumArray()

# Display the dimensions of the created array
print "Array dims:", imArr.shape

# Do something with the array...
imArr[:] = 0
radius, cx, cy = 64, 127, 164
y, x = np.ogrid[-radius: radius, 0 : radius]
index = x**2 + y**2 <= radius**2
imArr[cx-radius:cx+radius, cy-radius:cy+radius][index] = 255

# Call the "modified" method in order to update the viewer content
im1.modified()

