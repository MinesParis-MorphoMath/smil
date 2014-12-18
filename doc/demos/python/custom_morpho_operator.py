from smilPython import *

class myMeanMorphoFunction(MorphImageFunctionBase_UINT8):
  def processPixel(self, i, relOffList):
    pass
    pixSum = 0.
    for nb in relOffList:
      pixSum += self.imIn[i + nb]
    self.imOut[i] = int(pixSum / len(relOffList))


imIn = Image("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
imOut = Image(imIn)

func = myMeanMorphoFunction()
func(imIn, imOut)

imOut.show()

