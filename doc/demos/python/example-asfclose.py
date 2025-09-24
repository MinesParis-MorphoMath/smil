import smilPython as sp

imin = sp.Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")

imout = sp.Image(imin)

sp.asfClose(imin, imout, sp.SquSE(5))
