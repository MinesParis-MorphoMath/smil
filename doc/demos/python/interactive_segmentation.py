
from smilPython import *

class slot(EventSlot):
    def run(self, event):
        watershed(im2, v.getOverlay(), im3, im4)


im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/tools.png")
im2 = Image(im1)
im3 = Image(im1)
im4 = Image(im1)

gradient(im1, im2)

im1.show()
im3.show()
im4.showLabel()

v = im1.getViewer()

s = slot()
v.onOverlayModified.connect(s)
im1.show()

print "Now, right click on im1, in the \"Tools\" menu select \"Draw\", and draw markers (with different colors) on image."

# Will crash if not in a Qt loop
Gui.execLoop()
