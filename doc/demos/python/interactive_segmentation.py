from smilPython import *


im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/tools.png")
im2 = Image(im1)
im3 = Image(im1)
im4 = Image(im1)

gradient(im1, im2)

im1.show()
im3.show()
im4.showLabel()

class slot(EventSlot):
    def run(self, event):
        watershed(im2, v.getOverlay(), im3, im4)
s = slot()

v = im1.getViewer()
v.onOverlayModified.connect(s)

print "1) Right click on im1"
print "2) In the \"Tools\" menu select \"Draw\""
print "3) Draw markers (with different colors) on im1 and view the resulting segmentation"

# Will crash if not in a Qt loop
Gui.execLoop()
