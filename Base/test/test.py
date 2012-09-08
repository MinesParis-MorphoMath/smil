 
import sys
import time


from threading import Thread


sys.path.append(".")
sys.path.append("/home/faessel/src/ivp/faessel/")
sys.path.append("/home/mat/src/ivp/faessel/")

from smilPython import *
import mamba as mb


class testit(Thread):
   def __init__ (self):
      Thread.__init__(self)
      self.app = app = QtApp()
   def run(self):
      self.app._exec()

bench_sx = 1024
bench_sy = 40000
nruns = 5
     
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
  
  im1 = Image(10, 10)
  if read("/home/mat/src/morphee/trunk/utilities/Images/Gray/DNA_small.png", im1)==RES_ERR:
    read("/home/faessel/src/morphee/trunk/utilities/Images/Gray/DNA_small.png", im1)
  #if read("/home/mat/src/ivp/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0805_C22_3_20100326-105216/1.bmp", im1)==RES_ERR:
    #read("/home/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0805_C22_3_20100326-105216/1.bmp", im1)
  im2 = Image(im1)


  #se = sSE()
  se = hSE()



#im1 << 0

#im1.show()
#im1 = Image(50, 50)
#im1 = Image(50, 50)
#im1.show()

#im2.show()


def testBench(func=dilate, se=hSE(), binIm=False, prnt=1):
  if binIm:
    #tim1 = Image_bool(bench_sx, bench_sy)
    #tim2 = Image_bool(bench_sx, bench_sy)
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

def testBench16(func=dilate, se=hSE(), binIm=False, prnt=1):
  tim1 = Image_UINT16(bench_sx, bench_sy)
  tim2 = Image_UINT16(bench_sx, bench_sy)
  
  t1 = time.time()

  for i in range(int(nruns)):
    func(tim1, tim2, se)

  t2 = time.time()

  retval = (t2-t1)*1E3/nruns
  if prnt:
    print retval
  return retval

def testBench32(func=dilate, se=hSE(), binIm=False, prnt=1):
  tim1 = Image_UINT32(bench_sx, bench_sy)
  tim2 = Image_UINT32(bench_sx, bench_sy)
  
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

def bench_comp(new_sx=bench_sx, new_sy=bench_sy, new_nruns=nruns):
    global bench_sx, bench_sy, nruns
    bench_sx = new_sx
    bench_sy = new_sy
    nruns = new_nruns
    print "imSize:", bench_sx, ",", bench_sy
    print "\t\t\tMb\t\tSmil"
    print "dilate squ UINT8:\t", testBenchMb(mb.dilate, mb.sSE(1), 0, 0), "\t", testBench(dilate, sSE(), 0, 0)
    print "dilate squ UINT16:\t", "\t", "\t", testBench16(dilate, sSE(), 0, 0)
    print "dilate squ UINT32:\t", "\t", "\t", testBench32(dilate, sSE(), 0, 0)
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
  if read("/home/mat/src/ivp/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0805_C22_3_20100326-105216/1.bmp", im1)==RES_ERR:
    read("/home/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0805_C22_3_20100326-105216/1.bmp", im1)
    
  im2.setSize(im1)
  im3.setSize(im1)
  hMaxima(im1, 1, im2)
  im1.show()
  im2.show()


#im1 << "/home/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0805_C22_3_20100326-105216/1.bmp"
#im2.setSize(im1)
#im3.setSize(im1)
#im4 = Image(im1)

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

    
class myViewer(qtImageViewer_UINT8):
    def __init__(self):
      self.this = qtImageViewer_UINT8()
      self.this.own = 1
    def update(self):
      print "okii"
    def connect(self, im):
      imageViewer_UINT8.connect(self.this, im)
#class mySignal(Signal):
  #def __init(self):
    
#testBench()
#testInv()
#testMax()

#im1.show()
#enhanceContrast(im1, im1)


#imGrad = Image(im1)
#gradient(im1, imGrad)

#imMin = Image(im1)
#hMinima(imGrad, 15, imMin)

#imLbl = Image(im1, "UINT16")
#imLbl.setSize(im1)
#label(imMin, imLbl)
#imLbl.showLabel()

#imWs = Image(im1)
#imMark = Image(im1, "UINT16")
#imMark << 0
#imMark.setPixel(75, 40, 100)
#imMark.setPixel(120, 80, 200)
#imMark.showLabel()

#watershed(imGrad, imMark, imWs)
#imWs.show()

#class clA(classA):
  #def one(self):
    #print "python"

class slt(eventSlot):
  def __init__(self):
    #self.this = baseSlot()
    eventSlot.__init__(self)
  def run(self, ev):
    add(im1, 100, im2)
  #def two(self):
    #classA.two(self.pr)
  #def run(self, event=None):
     #print "one from python"
     
  #def _run(self, event=None):
     #self.this._run(event)

links.add(im1, open, im1, im2)

s = slt()
#im1.onModified.connect(s)
im1.modified()

     
#Core.getInstance().execLoop()

def ptrcast(t,ptr):
   import re
   if type(t)!=type(""):
      if len(t.__bases__)==1:
         t=t.__bases__[0].__name__
      else: t=t.__name__
   if hasattr(ptr,"this"):
      ptr=ptr.this
   return re.sub(r"(_p)+_\w+$","_p_"+t,ptr)

