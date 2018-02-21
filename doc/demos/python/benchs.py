from smilPython import *

sx = 1024
sy = 1024
bench_nruns = 1E3


# Load an image
imIn = Image("http://smil.cmm.mines-paristech.fr/images/DNA_small.png")

im1 = Image(sx, sy)
im2 = Image(im1)
im3 = Image(im1)

resize(imIn, im1)

print("*** Base ***")
bench(copy, im1, im2)
bench(fill,im2, 0)
bench(inv, im1, im2)
bench(add, im1, im2, im3)
bench(sub, im1, im2, im3)
bench(mul, im1, im2, im3)
bench(div, im1, im2, im3)

print("\n*** Arithmetic ***")
bench(inf, im1, im2, im3)
bench(sup, im1, im2, im3)
bench(equ, im1, im2, im3)
bench(low, im1, im2, im3)

print("\n*** Morphology ***")
bench(dilate, im1, im2, hSE(1))
bench(dilate, im1, im2, sSE(1))
bench(dilate, im1, im2, CrossSE(1))
bench(open, im1, im2, hSE(1))
bench(open, im1, im2, sSE(1))
bench(open, im1, im2, CrossSE(1))


