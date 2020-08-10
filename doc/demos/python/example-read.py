import smilPython as sp

img = sp.Image()
r = sp.getFileInfo("lena.jpg", img)

# Note, this is the same as :
img = sp.Image("lena.jpg")


