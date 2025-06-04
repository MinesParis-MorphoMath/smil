import smilPython as sp

img = sp.Image()
r = sp.read("lena.png", img)

# Note, this is the same as :
img = sp.Image("lena.png")
