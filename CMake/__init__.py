from .smil_Python import *
from .smilAdvancedPython import *
from .smilCorePython import *
from .smilBasePython import *
from .smilGuiPython import *
from .smilIOPython import *
from .smilMorphoPython import *


# -------------------------------------
# Functions renamed
#
def GeoDist(*args):
    """
    Function renamed. Use:
      r = distanceGeodesic(...)
    """
    return distanceGeodesic(*args)


def dist(*args):
    """
    Function renamed. Use:
      r = distance(...)
    """
    return distance(*args)


def dist_euclidean(*args):
    """
    Function renamed. Use:
      r = distanceEuclidean(...)
    """
    return distanceEuclidean(*args)


def fromNumArray(*args):
    """
    Function renamed. Use:
      r = fromNumpyArray(...)
    """
    return fromNumpyArray(*args)


def geoDil(*args):
    """
    Function renamed. Use:
      r = geoDilate(...)
    """
    return geoDilate(*args)


def geoEro(*args):
    """
    Function renamed. Use:
      r = geoErode(...)
    """
    return geoErode(*args)


def getNumArray(*args):
    """
    Function renamed. Use:
      r = getNumpyArray(...)
    """
    return getNumpyArray(*args)


def hFlip(*args):
    """
    Function renamed. Use:
      r = horizFlip(...)
    """
    return horizFlip(*args)


def labelWithMaxima(*args):
    """
    Function renamed. Use:
      r = labelWithMax(...)
    """
    return labelWithMax(*args)


def measAreas(*args):
    """
    Function renamed. Use:
      r = blobsArea(...)
    """
    return blobsArea(*args)


def measBarycenters(*args):
    """
    Function renamed. Use:
      r = blobsBarycenter(...)
    """
    return blobsBarycenter(*args)


def measBlobMoments(*args):
    """
    Function renamed. Use:
      r = blobsMoments(...)
    """
    return blobsMoments(*args)


def measBlobsEntropy(*args):
    """
    Function renamed. Use:
      r = blobsEntropy(...)
    """
    return blobsEntropy(*args)


def measBoundBoxes(*args):
    """
    Function renamed. Use:
      r = blobsBoundBox(...)
    """
    return blobsBoundBox(*args)


def measImageEntropy(*args):
    """
    Function renamed. Use:
      r = measEntropy(...)
    """
    return measEntropy(*args)


def measInertiaMatrices(*args):
    """
    Function renamed. Use:
      r = blobsMoments(...)
    """
    return blobsMoments(*args)


def measMaxVals(*args):
    """
    Function renamed. Use:
      r = blobsMaxVal(...)
    """
    return blobsMaxVal(*args)


def measMeanVals(*args):
    """
    Function renamed. Use:
      r = blobsMeanVal(...)
    """
    return blobsMeanVal(*args)


def measMedianVal(*args):
    """
    Function renamed. Use:
      r = medianVal(...)
    """
    return medianVal(*args)


def measMedianVals(*args):
    """
    Function renamed. Use:
      r = blobsMedianVal(...)
    """
    return blobsMedianVal(*args)


def measMinVals(*args):
    """
    Function renamed. Use:
      r = blobsMinVal(...)
    """
    return blobsMinVal(*args)


def measModeVal(*args):
    """
    Function renamed. Use:
      r = modeVal(...)
    """
    return modeVal(*args)


def measModeVals(*args):
    """
    Function renamed. Use:
      r = blobsModeVal(...)
    """
    return blobsModeVal(*args)


def measRangeVals(*args):
    """
    Function renamed. Use:
      r = blobsRangeVal(...)
    """
    return blobsRangeVal(*args)


def measVolumes(*args):
    """
    Function renamed. Use:
      r = blobsVolume(...)
    """
    return blobsVolume(*args)


def stretchHist(*args):
    """
    Function renamed. Use:
      r = stretchHistogram(...)
    """
    return stretchHistogram(*args)


def vFlip(*args):
    """
    Function renamed. Use:
      r = vertFlip(...)
    """
    return vertFlip(*args)


def valueLists(*args):
    """
    Function renamed. Use:
      r = blobsValueList(...)
    """
    return blobsValueList(*args)


# -------------------------------------
# Shortcuts
#
def getDefaultSE(*args):
    """
    Shortcut
      r = Morpho.getDefaultSE(...)
    """
    return Morpho.getDefaultSE(*args)


def setDefaultSE(*args):
    """
    Shortcut
      r = Morpho.setDefaultSE(...)
    """
    return Morpho.setDefaultSE(*args)


# -------------------------------------
# Additions
#


#
#
def colorGradientHLS(imIn, se=Morpho.getDefaultSE(), convertFirstToHLS=True):
    imOut = Image(imIn, imtype="UINT8")
    r = gradientHLS(imIn, imOut, se, convertFirstToHLS)
    if r == 1:
        return imOut
    return None


#
#
def colorGradientLAB(imIn, se=Morpho.getDefaultSE(), convertFirstToLAB=True):
    imOut = Image(imIn, imtype="UINT8")
    r = gradientLAB(imIn, imOut, se, convertFirstToLAB)
    if r == 1:
        return imOut
    return None
