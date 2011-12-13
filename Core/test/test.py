 
import sys
import time

from smilPython import *

from threading import Thread


sys.path.append("/home/faessel/src/ivp/faessel/")


class testit(Thread):
   def __init__ (self):
      Thread.__init__(self)
      self.app = app = QtApp()
   def run(self):
      self.app._exec()

bench_sx = 1000
bench_sy = 1000
     
if ('im1' in locals())==0:
  #app = QtGui.QApplication(sys.argv)
  #im1 = Image_UINT8(1024, 1024)
  #tapp = QtApp()
  im1 = Image(bench_sx, bench_sy)
  im2 = Image(im1)
  im3 = Image(im1)
  #app = QtApp()
  #app._exec()
  #im1.show()

  #se = sSE()
  se = hSE()



im1 << 0

#im1.show()
#im2.show()


def testBench(binIm=False):
  if binIm:
    tim1 = Image_bool(bench_sx, bench_sy)
    tim2 = Image_bool(bench_sx, bench_sy)
  else:
    tim1 = Image_UINT8(bench_sx, bench_sy)
    tim2 = Image_UINT8(bench_sx, bench_sy)
  
  tse = hSE()
  
  nruns = 1E3 # 5E3
  t1 = time.time()

  for i in range(int(nruns)):
    dilate(tim1, tim2, tse)
    #addIm(im1, im2, im3)
    #supIm(im1, im2, im3)

  t2 = time.time()

  print (t2-t1)*1E3/nruns

def testBenchMb(binIm=False):
  import mamba as mb
  if (binIm):
    mIm1 = mb.imageMb(1)
    mIm2 = mb.imageMb(1)
  else:
    mIm1 = mb.imageMb()
    mIm2 = mb.imageMb()
  
  mIm1.setSize(bench_sx, bench_sy)
  mIm2.setSize(bench_sx, bench_sy)
  
  mse = mb.hSE(1)
  
  nruns = 1E3 # 5E3
  t1 = time.time()

  for i in range(int(nruns)):
    mb.dilate(mIm1, mIm2, mse)
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
  
def testMax():
  im1 << "/home/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0805_C22_3_20100326-105216/1.bmp"
  im2.setSize(im1)
  im3.setSize(im1)
  hMaxima(im1, 1, im2)
  im1.show()
  im2.show()


im1 << "/home/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0805_C22_3_20100326-105216/1.bmp"
im2.setSize(im1)
im3.setSize(im1)
im4 = Image(im1)

def levels(imIn, imOut):
    s = imIn.getSize()
    tmpIm = Image(imIn)
    tmpIm2 = Image(imIn)
    centerIm = Image(imIn)
    centerIm << 0
    centerIm.setPixel(255, s[0]/2, s[1]/2)
    dilate(centerIm, centerIm, hSE(2))
    
    imOut << 0
    
    l1 = 50
    l2 = 200
    
    thresh(imIn, 0, l1, tmpIm)
    fillHoles(tmpIm, tmpIm2)
    build(centerIm, tmpIm2, tmpIm)
    test(tmpIm, l1, 0, imOut)
    
    thresh(imIn, l1, l2, tmpIm)
    build(centerIm, tmpIm, tmpIm2)
    fillHoles(tmpIm2, tmpIm)
    test(tmpIm, l2, 0, tmpIm2)
    sup(imOut, tmpIm2, imOut)
    
    thresh(imIn, l2, 255, tmpIm2)
    inf(tmpIm, tmpIm2, tmpIm)
    sup(imOut, tmpIm, imOut)

    
#testBench()
#testInv()
#testMax()

