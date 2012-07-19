from smilPython import *

imIn = Image("http://cmm.ensmp.fr/~faessel/smil/images/DNA_small.png")
imIn.show()

imGrad = Image(imIn)
gradient(imIn, imGrad)

imMin = Image(imIn)
hMinima(imGrad, 20, imMin)

imMark = Image(imIn, "UINT16")
imMark << 0
imMark.setPixel(75, 40, 1)
imMark.setPixel(78, 86, 2)
imMark.setPixel(88, 76, 3)
dilate(imMark, imMark, 2)

imWS = Image(imIn)
watershed(imGrad, imMark, imWS)

imWS.show()
imIn.getViewer().drawOverlay(imWS & 1)

