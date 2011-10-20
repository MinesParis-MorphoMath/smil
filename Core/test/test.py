 
import sys
import time

from smil import *


from threading import Thread

class testit(Thread):
   def __init__ (self):
      Thread.__init__(self)
      self.app = app = QtApp()
   def run(self):
      self.app._exec()


     
if ('im1' in locals())==0:
  #app = QtGui.QApplication(sys.argv)
  #im1 = Image_UINT8(1024, 1024)
  #tapp = QtApp()
  im1 = Image(1000, 1000)
  im2 = Image(im1)
  im3 = Image(im1)
  #app = QtApp()
  #app._exec()
  #im1.show()

  se = hSE()



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
  im2.setSize(im1)
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
