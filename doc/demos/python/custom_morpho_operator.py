from smilPython import *

class myMeanMorphoFunction(MorphImageFunctionBase_UINT8):
  def processPixel(self, i, relOffList):
    pixSum = 0.
    for nb in relOffList:
      pixSum += self.imageIn[i + nb]
    self.imageOut[i] = int(pixSum / len(relOffList))


imIn = Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
imOut = Image(imIn)

func = myMeanMorphoFunction()
func(imIn, imOut)

imOut.show()

