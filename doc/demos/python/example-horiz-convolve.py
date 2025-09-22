import smilPython as sp

imIn = sp.Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
imOut = sp.Image(imIn)
imIn.show()
imOut.show()

kernel = [0.0545, 0.2442, 0.4026, 0.2442, 0.0545]
sp.horizConvolve(imIn, kernel, imOut)
