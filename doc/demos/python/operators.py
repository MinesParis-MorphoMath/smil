from smilPython import *
import time

# Load an image
im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
#im1.show()

im2 = Image(im1)
# Mask of im1>0:
# im1 > 100 returns an image = 255 when im1>100 and = 0 otherwise
# take the result and make the inf (&) with the original image

nbr_runs = 1E3

t1 = time.time()

i = 0.
while i<nbr_runs:
    im2 << ( (im1>100) & im1 )
    #im2.show()
    i += 1

t2 = time.time()
print (t2-t1)*1E3/nbr_runs

im3 = Image(im1)
sePts = ((0,0), (0,1), (1,1), (1,0), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1))
# Fill 0
im3 << 0
# Take the sup with each translation of im1...
for (dx,dy) in sePts:
    im3 |= trans(im1, dx, dy)
#im3.show()
