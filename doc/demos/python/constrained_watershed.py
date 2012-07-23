from smilPython import *

# Load an image
imIn = Image("http://cmm.ensmp.fr/~faessel/smil/images/DNA_small.png")
imIn.show()

# Create a gradient image
imGrad = Image(imIn)
gradient(imIn, imGrad)

# Manually impose markers on image
imMark = Image(imIn, "UINT16")
imMark << 0
# One for the background...
imMark.setPixel(75, 40, 1)
# and one on two connected particules
imMark.setPixel(78, 86, 2)
imMark.setPixel(88, 76, 3)

# Dilate the markers to avoid to be blocked in a minimum
dilate(imMark, imMark, 2)
imMark.showLabel()

# Create the watershed
imWS = Image(imIn)
watershed(imGrad, imMark, imWS)

# Display output
imWS.show()

# Display the output as overlay on the original image
imIn.getViewer().drawOverlay(imWS & 1)

