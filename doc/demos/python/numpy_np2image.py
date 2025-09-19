import smilPython as sp
import numpy as np

# create a 10x10 NumPy array
ar = np.zeros((32, 32), "uint8")
ar[8, :] = 255
ar[:, 16] = 255

# creates a Smil image and set it's content to "ar"
img = sp.Image(ar)
