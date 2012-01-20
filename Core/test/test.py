 
import sys
import time

from smilPython import *

from threading import Thread


sys.path.append("/home/faessel/src/ivp/faessel/")
sys.path.append("/home/mat/src/ivp/faessel/")

import mamba as mb


class testit(Thread):
   def __init__ (self):
      Thread.__init__(self)
      self.app = app = QtApp()
   def run(self):
      self.app._exec()

bench_sx = 1024*5
bench_sy = 1024
nruns = 1E2
     
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


def testBench(func=dilate, se=hSE(), binIm=False, prnt=1):
  if binIm:
    tim1 = Image_Bit(bench_sx, bench_sy)
    tim2 = Image_Bit(bench_sx, bench_sy)
  else:
    tim1 = Image_UINT8(bench_sx, bench_sy)
    tim2 = Image_UINT8(bench_sx, bench_sy)
  
  t1 = time.time()

  for i in range(int(nruns)):
    func(tim1, tim2, se)

  t2 = time.time()

  retval = (t2-t1)*1E3/nruns
  if prnt:
    print retval
  return retval

def testBenchMb(func=dilate, se=mb.hSE(1), binIm=False, prnt=1):
  if (binIm):
    mIm1 = mb.imageMb(1)
    mIm2 = mb.imageMb(1)
  else:
    mIm1 = mb.imageMb()
    mIm2 = mb.imageMb()
  
  mIm1.setSize(bench_sx, bench_sy)
  mIm2.setSize(bench_sx, bench_sy)
  
  t1 = time.time()

  for i in range(int(nruns)):
    func(mIm1, mIm2, se)

  t2 = time.time()
  
  retval = (t2-t1)*1E3/nruns
  if prnt:
    print retval
  return retval

def bench_comp():
    print "imSize:", bench_sx, ",", bench_sy
    print "\t\t\tMb\t\tSmil"
    print "dilate squ UINT8:\t", testBenchMb(mb.dilate, mb.sSE(1), 0, 0), "\t", testBench(dilate, sSE(), 0, 0)
    print "dilate hex UINT8:\t", testBenchMb(mb.dilate, mb.hSE(1), 0, 0), "\t", testBench(dilate, hSE(), 0, 0)
    print "dilate squ bin:\t\t", testBenchMb(mb.dilate, mb.sSE(1), 1, 0), "\t", testBench(dilate, sSE(), 1, 0)
    print "dilate hex bin:\t\t", testBenchMb(mb.dilate, mb.hSE(1), 1, 0), "\t", testBench(dilate, hSE(), 1, 0)
    
    print "erode squ UINT8:\t", testBenchMb(mb.erode, mb.sSE(1), 0, 0), "\t", testBench(erode, sSE(), 0, 0)
    print "erode hex UINT8:\t", testBenchMb(mb.erode, mb.hSE(1), 0, 0), "\t", testBench(erode, hSE(), 0, 0)
    print "erode squ bin:\t\t", testBenchMb(mb.erode, mb.sSE(1), 1, 0), "\t", testBench(erode, sSE(), 1, 0)
    print "erode hex bin:\t\t", testBenchMb(mb.erode, mb.hSE(1), 1, 0), "\t", testBench(erode, hSE(), 1, 0)
    
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

def run():
    tim1 = Image_Bit(bench_sx, bench_sy)
    tim2 = Image_Bit(bench_sx, bench_sy)
    dilate(tim1, tim2, hSE())

#testBench()
#testInv()
#testMax()

