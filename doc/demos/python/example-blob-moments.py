import smilPython as sp

# How to print moments
def printBlobMoments(m, title=""):
    print(title.center(90))
    hdr = ["m00", "m10", "m01", "m11", "m20", "m02"]
    s = " " * 5
    for h in hdr:
        s += h.rjust(16)
    print(s)
    for label in m.keys():
        s = '{:3} :'.format(label)
        for v in m[label]:
            s += '  {:14.2f}'.format(v)
        print(s)

# Serious work begins here
#
binaryImage = False
if binaryImage:
    imageName = "http://smil.cmm.mines-paristech.fr/images/balls.png"
    imo = sp.Image(imageName)
    iml = sp.Image(imo, 'UINT16')
    # normalize binary image values
    imo /= 255
    sp.label(imo, iml)
else:
    imageName = "http://smil.cmm.mines-paristech.fr/images/tools.png"
    imo = sp.Image(imageName)
    imThr = sp.Image(imo)
    sp.topHat(imo, imThr, sp.hSE(20))
    sp.threshold(imThr, imThr)
    iml = sp.Image(imo, "UINT16")
    sp.label(imThr, iml)

# create blobs
blobs = sp.computeBlobs(iml)

print("=== > Image : ", imageName, "\n")

# Calculate not centered moments
moments = sp.measBlobMoments(imo, blobs, False)
printBlobMoments(moments, "Not centered moments")

print("=" * 102)
# Calculate centered moments
moments = sp.measBlobMoments(imo, blobs, True)
printBlobMoments(moments, "Centered moments")
