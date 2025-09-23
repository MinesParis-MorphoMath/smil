from smilPython import *
from math import *

# Load an image
imIn = Image("https://smil.cmm.minesparis.psl.eu/images/tools.png")
imIn.show()

imThr = Image(imIn)
topHat(imIn, imThr, hSE(20))
threshold(imThr, imThr)

imLbl = Image(imIn, "UINT16")
label(imThr, imLbl)
imLbl.showLabel()


def fitRectangle(mat):
    m00, m10, m01, m11, m20, m02 = mat

    if m00 == 0:
        return 0, 0, 0, 0, 0

    # COM
    xc = int(m10 / m00)
    yc = int(m01 / m00)

    # centered matrix (central moments)
    u00 = m00
    u20 = m20 - m10**2 / m00
    u02 = m02 - m01**2 / m00
    u11 = m11 - m10 * m01 / m00

    # eigen values
    delta = 4 * u11**2 + (u20 - u02) ** 2
    I1 = (u20 + u02 + sqrt(delta)) / 2
    I2 = (u20 + u02 - sqrt(delta)) / 2

    theta = 0.5 * atan2(-2 * u11, (u20 - u02))

    # Equivalent rectangle
    # I1 = a**2 * S / 12, I2 = b**2 * S / 12
    a = int(sqrt(12 * I1 / u00))
    b = int(sqrt(12 * I2 / u00))

    return xc, yc, a, b, theta


# Compute Blobs
blobs = computeBlobs(imLbl)

# Compute Inertia Matrices

mats = blobsMoments(imIn, blobs)
bboxes = blobsBoundBox(imLbl)
imDraw = Image(imIn)

print("{:5s}      {:>5s} {:>5s} {:>6s}".format("Label", "A", "B", "Theta"))
for b in blobs.keys():
    mat = xc, yc, A, B, theta = fitRectangle(mats[b])
    # print(str(b) + "\t" + str(A) + "\t" + str(B) + "\t" + str(theta))
    print(f"{b:5d}      {A:>5d} {B:>5d} {theta:6.3f}")
    dx = A / 2 * cos(pi - theta)
    dy = A / 2 * sin(pi - theta)
    drawLine(imDraw, int(xc - dx), int(yc - dy), int(xc + dx), int(yc + dy), b)
    dx = B / 2 * sin(theta)
    dy = B / 2 * cos(theta)
    drawLine(imDraw, int(xc - dx), int(yc - dy), int(xc + dx), int(yc + dy), b)


print("on Overlay")
imIn.getViewer().drawOverlay(imDraw)

print("after overlay")
input()
