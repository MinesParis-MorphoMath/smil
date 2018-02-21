smilCoreOctave
smilGuiOctave

im1 = Image_UINT8("http://smil.cmm.mines-paristech.fr/images/lena.png")
im1.show()
im2 = Image_UINT8(im1)
im2.show()

Gui.execLoop()
