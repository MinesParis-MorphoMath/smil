
import smilPython as sp

imIn = sp.Image("http://smil.cmm.mines-paristech.fr/images/lena.png")
imOut = sp.Image(imIn)
imIn.show()
imOut.show()

sp.gaussianFilter(imIn, 3, imOut)

