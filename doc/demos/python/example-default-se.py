
import smilPython as sp

im = sp.Image("lena.png")

# set new default StrElt and save old one
saveSE = sp.Morpho.getDefaultSE()
sp.Morpho.setDefaultSE(sp.SquSE))

# do something
imd = sp.Image(im)
sp.dilate(im, imd, 3)

# restore old default StrElt
sp.Morpho.setDefaultSE(saveSE)

