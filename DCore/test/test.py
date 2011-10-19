 
from PyQt4 import QtGui, QtCore
import sys
import time

from smilCorePython import *
import smilCorePython as sc


from threading import Thread

class testit(Thread):
   def __init__ (self):
      Thread.__init__(self)
      self.app = app = QtApp()
   def run(self):
      self.app._exec()


      
import gc, sys
def find_names(obj):
  frame = sys._getframe()
  for frame in iter(lambda: frame.f_back, None):
      frame.f_locals
  result = []
  for referrer in gc.get_referrers(obj):
      if isinstance(referrer, dict):
	  for k, v in referrer.iteritems():
	      if v is obj:
		  result.append(k)
  return result

def showImage(img):
  imName = find_names(img)[1]
  img.setName(imName)
  img.show()
     
      
if ('app' in locals())==0:
  app = QtGui.QApplication(sys.argv)
  #im1 = Image_UINT8(1024, 1024)
  #tapp = QtApp()
  im1 = Image_UINT8(1000, 1000)
  im2 = Image_UINT8(im1)
  im3 = Image_UINT8(im1)
  #app = QtApp()
  #app._exec()
  #im1.show()

  se = hSE()


createImageFunctions = ( Image_UINT8, Image_UINT16 )

def Image(*args):
    argNbr = len(args)
    argTypeStr = [ str(type(a)) for a in args ]
    
    if argTypeStr[0].rfind("Image_")!=-1:
      srcIm = args[0]
      if argNbr==1:
	return createImage(srcIm)
    else:
	return createImageFunctions[args[1]](srcIm.getWidth(), srcIm.getHeight(), srcIm.getDepth())
    if args==():
	return Image_UINT8()
	

im1 << 0

#im1.show()
#im2.show()

nruns = 1E3 # 5E3
t1 = time.time()

for i in range(int(nruns)):
  dilate(im1, im2, se)
  #addIm(im1, im2, im3)
  #supIm(im1, im2, im3)

t2 = time.time()

print (t2-t1)*1E3/nruns


def testInv():
  im1.setSize(50,50)
  im1 << 127
  im1.show()
  for i in range(50):
      im1.setPixel(255, 25, i)
  im2 = createImage(im1)
  inv(im1, im2)
  im2.show()

testInv()

#def dilate(imIn, imOut, se=sSE()):
  #tmpIm = Image(imIn)
  #copyIm(imIn, tmpIm)
  #for s in range(se.size):
    #dilIm(tmpIm, imOut, se)
    #copyIm(imOut, tmpIm)
    
#im1.setSize(40, 40)
#im2.setSize(40, 40)
#im1 << 100
##im2 << 0



#im1 << "birds_gradmosa.png"
#im2 = Image(im1)
#im1.show()
#dilateIm(im1, im2, hSE(10))
#im2.show()
