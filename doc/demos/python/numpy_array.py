from smilPython import *

# Create an image
im1 = Image(512,512)
im1.show()

# Create a numpy array containing the real image pixels
array = im1.getNumArray()

# Make something with the array (fill the image with the value 127)
array[:] = 127

# Transform the array into a 2D array (transposed, to have the right image orientation)
array_2d = array.reshape(512,512).transpose()

# Make something with the 2D array (draw a vectical segment of lenght 128)
array_2d[128, 128:256] = 255

# Call the "modified" method in order to update the viewer content
im1.modified()

