import smilPython as sp
import numpy as np
from numpy import linalg as la

# get image
imIn = sp.Image("http://smil.cmm.mines-paristech.fr/images/tools.png")

# labelize input image
imThr = sp.Image(imIn)
sp.topHat(imIn, imThr, sp.hSE(20))
sp.threshold(imThr, imThr)
imLbl = sp.Image(imIn, "UINT16")
sp.label(imThr, imLbl)

# compute blobs and get their data
blobs   = sp.computeBlobs(imLbl)
central = True
inertia = sp.blobsInertiaMatrix(imIn, blobs, central)
barys   = sp.blobsBarycenter(imIn, blobs)

nshape = (2, 2)
if imIn.getDimension() == 3:
  nshape = (3, 3)

for k in inertia.keys():
    print("=" * 64)
    s = ""
    for v in barys[k]:
        s += " {:6.1f}".format(v)
    print("Blob label : {:3d} - Barycenter :{:}".format(k, s))
    # Smil returns inertia matrix as a vector. Reshape it.
    xi = np.array(inertia[k])
    xi = xi.reshape(nshape)
    np.set_printoptions(precision=3)
    print("    ====== Inertia Matrix")
    print(xi)
    # let numpy evaluate eingenvalues and eingenvectors
    w, v = la.eig(xi)
    print("    ====== Eingenvalues")
    print(w)
    print("    ====== Eingenvectors")
    print(v)

