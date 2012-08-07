from smilPython import *

im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")

im2 = Image(im1)

im2 << ~im1

im2.show()
