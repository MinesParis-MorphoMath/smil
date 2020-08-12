
import smilpython as sp

imin = sp.Image("http://smil.cmm.mines-paristech.fr/images/lena.png")

imout = sp.Image(imin)

sp.asfClose(imin, imout, sp.SquSE(5))
