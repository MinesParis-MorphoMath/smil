import smilPython as sp
import time

thresh = 1000

im = sp.Image("http://smil.cmm.mines-paristech.fr/images/balls.png")
iml = sp.Image(im)
img = sp.Image(im)
ims = sp.Image(im)

sp.label(im, iml)

im.show("balls.png")
iml.showLabel("iml")

sp.areaThreshold(iml, thresh, True, img)
img.showLabel("img")
sp.areaThreshold(im, thresh, False, ims)
ims.show("ims")

nlold = 0
for threshold in range(1, 6000, 20):
    sp.areaThreshold(im, threshold, True, ims)
    nl = sp.label(ims, iml)
    if nl != nlold:
        print(' Threshold {:6d} : {:3d} blobs'.format(threshold, nl))
        sp.Gui.processEvents()
        time.sleep(1)
    if nl == 0:
        break
    nlold = nl
