smilCoreOctave
smilGuiOctave

im1 = Image_UINT8("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
im1.show()
im2 = Image_UINT8(im1)
im2.show()

Gui.execLoop()
