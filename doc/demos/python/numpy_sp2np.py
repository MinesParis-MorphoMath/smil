
import smilPython as sm
import numpy as np

# read a PNG image file (8 bits gray image)
file = "my-image.png")
img = sm.Image(file)

# show the image
img.show(file)

# get a NumPy array
p = img.getNumArray()

# let's threshold the image
t = 127
p[p >= t] = 255
p[p <  t] = 0

# Call the "modified" method in order to update the viewer content
img.modified()

