import smilPython as sp

imIn = sp.Image(10, 10, 10)
imOut = sp.Image(imIn)
imTmp = sp.Image(imIn)

# rotation 90 degres around Y axis
# 1. exchange axes y and z
sp.matTranspose(imIn, imTmp, "xzy")
# 2. rotate image around new z axis
sp.rotateX90(imTmp, 1, imTmp)
# 3. put axis y back
sp.matTranspose(imTmp, imOut, "xzy")
