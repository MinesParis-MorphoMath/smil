from smilPython import *

imIn = Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
imGrad = Image(imIn)
imMark = Image(imIn, "UINT16")
imBasins = Image(imMark)

gradient(imIn, imGrad)
hMinimaLabeled(imGrad, 5, imMark)

nRegions = 25
extType = "v"

# Graph version

imGraphOut = Image(imMark)
g = watershedExtinctionGraph(imGrad, imMark, imBasins, extType)
g.removeLowEdges(nRegions)
graphToMosaic(imBasins, g, imGraphOut)
# Re-labelize (usefull only to have the same label values in both versions)
label(imGraphOut, imGraphOut)
imGraphOut.showLabel()


# Image version

imImgOut = Image(imMark)
watershedExtinction(imGrad, imMark, imImgOut, imBasins, extType)
compare(imImgOut, ">", nRegions, 0, imMark, imMark)
basins(imGrad, imMark, imImgOut)
# Re-labelize (usefull only to have the same label values in both versions)
label(imImgOut, imImgOut)
imImgOut.showLabel()
