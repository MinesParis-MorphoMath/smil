import smilPython as sp

im = sp.Image("https://smil.cmm.minesparis.psl.eu/images/balls.png")
imLbl1 = sp.Image(im, "UINT16")
imLbl2 = sp.Image(imLbl1)

sp.label(im, imLbl1)

# We can use a Smil Map
# lookup = Map_UINT16_UINT16()
# or directly a python dict
lookup = dict()
lookup[1] = 2
lookup[5] = 3
lookup[2] = 1

sp.applyLookup(imLbl1, lookup, imLbl2)

imLbl1.showLabel()
imLbl2.showLabel()
