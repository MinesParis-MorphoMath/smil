smilOctave
smilCoreOctave
smilGuiOctave
smilIOOctave
smilBaseOctave
smilMorphoOctave


im1 = Image_UINT8("http://smil.cmm.mines-paristech.fr/images/lena.png")
im2 = Image_UINT8(im1)
dilate(im1, im2, hSE(5))

im1.show()
im2.show()
Gui.execLoop()
