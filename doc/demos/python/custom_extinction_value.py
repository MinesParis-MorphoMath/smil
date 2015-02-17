from smilPython import *

class myAreaExtinction(ExtinctionFlooding_UINT8_UINT16):
  def createBasins(self, nbr):
    self.areas = [0]*nbr
    # Call parent class method
    ExtinctionFlooding_UINT8_UINT16.createBasins(self, nbr)
  def insertPixel(self, offset, lbl):
    self.areas[lbl] += 1
  def mergeBasins(self, lbl1, lbl2):
    if self.areas[lbl1] > self.areas[lbl2]:
      eater = lbl1
      eaten = lbl2
    else:
      eater = lbl2
      eaten = lbl1
    self.extinctionValues[eaten] = self.areas[eaten]
    self.areas[eater] += self.areas[eaten]
    return eater
  def finalize(self, lbl):
    self.extinctionValues[lbl] += self.areas[lbl]


imIn = Image("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
imGrad = Image(imIn)
imMark = Image(imIn, "UINT16")
imExtRank = Image(imIn, "UINT16")

gradient(imIn, imGrad)
hMinimaLabeled(imGrad, 25, imMark)

aExt = myAreaExtinction()
aExt.floodWithExtRank(imIn, imMark, imExtRank)

imExtRank.showLabel()

