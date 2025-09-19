# import smilPython module
import smilPython as sp

# define which structuring element to use
se = sp.SquSE()

# get your first image
# try replacing the URL by some image stored in your computer.
imIn = sp.Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
imIn.show("Lena original")

# declare two images for results
imA = sp.Image(imIn)
imA.show("erode")

imB = sp.Image(imIn)
imB.show("dilate")

# erosion and dilation
r = sp.erode(imIn, imA, se())
r = sp.dilate(imIn, imB, se())

input("Hit enter to continue")

# now let's do some filtering with a SE of size 2
r = sp.open(imIn, imA, se(2))
imA.setName("open")
r = sp.close(imIn, imB, se(2))
imB.setName("close")

input("Hit enter to continue")
