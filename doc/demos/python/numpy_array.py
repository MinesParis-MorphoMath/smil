
import smilPython as sp
import numpy as np

# Create an image
sx = 256
sy = 384
im1 = sp.Image(sx, sy)
im1.show()

# Create a numpy array containing the real image pixels
imArr = im1.getNumpyArray()

# Display the dimensions of the created array
print("Array dims:", imArr.shape)

# Do something with the array... E.g., draw a circle
imArr[:] = 0
# the circle will be centered at the center of the image
radius, cx, cy = 64, sy//2, sx//2
y, x = np.ogrid[0:sx, 0:sy]
# get the indexes of the pixels inside the circle
index = (x - cx)**2 + (y - cy)**2 <= radius**2
imArr[:,:][index] = 255

# Call the "modified" method in order to update the viewer content
im1.modified()

input()

