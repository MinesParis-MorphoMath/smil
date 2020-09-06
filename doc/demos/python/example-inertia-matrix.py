import smilPython as sp
import numpy as np

imIn = sp.Image("http://smil.cmm.mines-paristech.fr/images/tools.png")

imThr = sp.Image(imIn)
sp.topHat(imIn, imThr, sp.hSE(20))
sp.threshold(imThr, imThr)
imLbl = sp.Image(imIn, "UINT16")
sp.label(imThr, imLbl)

blobs = sp.computeBlobs(imLbl)

central = True

barys = sp.measBarycenters(imIn, blobs)
inertia = sp.inertiaMatrices(imIn, blobs, central)

nshape = (2, 2)
if imIn.getDimension() == 3:
  nshape = (3, 3)

from numpy import linalg as la

for k in inertia.keys():
    print("=" * 64)
    s = ""
    for v in barys[k]:
        s += " {:6.1f}".format(v)
    print("Blob label : {:3d} - Barycenter :{:}".format(k, s))
    xi = np.array(inertia[k])
    xi = xi.reshape(nshape)
    np.set_printoptions(precision=3)
    print("    ====== Inertia Matrix")
    print(xi)
    w, v = la.eig(xi)
    print("    ====== Eingenvalues")
    print(w)
    print("    ====== Eingenvectors")
    print(v)
