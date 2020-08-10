import smilPython as sp

fname = "imageraw.raw"

type = 'UINT16'
width = 256
height = 384
depth = 512

img = sp.Image(type)
sp.readRAW(fname, width, height, depth, img)

