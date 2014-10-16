from smilPython import *

#im = Image ("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
im = Image ("/home/chabardes/src/samg/trunk/chabardes/crop_300_OK.vtk")

imG = Image (im)
out = Image (im)
out2 = Image (im)
imD = Image (im)

se = Cross3DSE()

gradient (im, imG, se)
fastMinima (imG, out, se)
minima (imG, out2, se)
diff (out, out2, imD)

imD.show ()


#benchmarking
bench (fastMinima, imG, out, se, nbr_runs=10)
bench (minima, imG, out2, se, nbr_runs=10)
