
import smilPython as sp

imIn = sp.Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
imOut = sp.Image(imIn)
imIn.show()
imOut.show()

sp.gaussianFilter(imIn, 3, imOut)

