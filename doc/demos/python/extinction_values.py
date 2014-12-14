from smilPython import *

imIn = Image("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
imGrad = Image(imIn)
imMark = Image(imIn, "UINT16")
imBasins = Image(imMark)

gradient(imIn, imGrad)
hMinimaLabeled(imGrad, 5, imMark)

nRegions = 25

# Graph version

imGraphOut = Image(imMark)
g = watershedExtinctionGraph(imGrad, imMark, imBasins, "v")
g.removeLowEdges(nRegions)
graphToMosaic(imBasins, g, imGraphOut)
# Re-labelize (usefull only to have the same label values in both versions)
label(imGraphOut, imGraphOut)
imGraphOut.showLabel()


# Image version

imImgOut = Image(imMark)
watershedExtinction(imGrad, imMark, imImgOut, imBasins, "v")
compare(imImgOut, ">", nRegions, 0, imMark, imMark);
basins(imGrad, imMark, imImgOut)
# Re-labelize (usefull only to have the same label values in both versions)
label(imImgOut, imImgOut)
imImgOut.showLabel()
