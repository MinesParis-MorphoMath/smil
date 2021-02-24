
import smilPython as sm
import numpy as np

# create a 10x10 NumPy array
a = np.array(range(100), 'uint8')
a = a.reshape(10, 10)

# creates a Smil image and set it's content to "a"
img = sm.Image()
img.fromNumArray(a)

