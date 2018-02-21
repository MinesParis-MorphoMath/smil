from smilPython import *

class myWSFlooding(WatershedFlooding_UINT8_UINT16):
  def initialize(self, imIn, imLbl, imOut, se):
    # Call parent class method
    WatershedFlooding_UINT8_UINT16.initialize(self, imIn, imLbl, imOut, se)
    self.imgWS.updatesEnabled = True
    self.imgLbl.updatesEnabled = True
    self.nbrPixProcessed = 0
    self.refresh_every = 10 # refresh every n pixels processed
    return RES_OK
  def processPixel(self, offset):
    # Call parent class method
    WatershedFlooding_UINT8_UINT16.processPixel(self, offset)
    if self.nbrPixProcessed>=self.refresh_every:
        self.imgWS.modified()
        self.imgLbl.modified()
        Gui.processEvents()
        self.nbrPixProcessed = 0
    else:
        self.nbrPixProcessed += 1


if not "imIn" in globals():
  imIn = Image("http://smil.cmm.mines-paristech.fr/images/lena.png")
  imGrad = Image(imIn)
  imWS = Image(imIn)
  imMark = Image(imIn, "UINT16")
  imBasins = Image(imIn, "UINT16")

gradient(imIn, imGrad)
hMinimaLabeled(imGrad, 10, imMark)

imGrad.show()
imWS.showLabel()
imBasins.showLabel()

wsFlood = myWSFlooding()
wsFlood.flood(imIn, imMark, imWS, imBasins, sSE())




