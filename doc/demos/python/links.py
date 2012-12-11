from smilPython import *
import time

# Load an image
imIn= Image("http://cmm.ensmp.fr/~faessel/smil/images/DNA_small.png")
imThresh = Image(imIn)
imDist = Image(imIn)

imIn.show()
imThresh.show()
imDist.showLabel()

def displMax():
  print "Distance max value:", rangeVal(imDist)[1]

links = linkManager()
links.add(imIn, threshold, imIn, 255, imThresh)
links.add(imThresh, dist, imThresh, imDist)
links.add(imDist, displMax)

for i in range(1, 10):
  print "\nThreshold level:", i*10
  links[0].args[1] = i*10
  time.sleep(1)
