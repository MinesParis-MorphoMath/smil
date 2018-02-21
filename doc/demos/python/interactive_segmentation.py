from smilPython import *


im1 = Image("http://smil.cmm.mines-paristech.fr/images/tools.png")
im2 = Image(im1)
im3 = Image(im1)
im4 = Image(im1)
imOverl = Image(im1)

gradient(im1, im2)

im1.show()
im3.show()
im4.showLabel()

v = im1.getViewer()

class slot(EventSlot):
    def run(self, event=None):
      v.getOverlay(imOverl)
      watershed(im2, imOverl, im3, im4)
        
s = slot()

v.onOverlayModified.connect(s)
v.onOverlayModified.trigger()


print("1) Right click on im1")
print("2) In the \"Tools\" menu select \"Draw\"")
print("3) Draw markers (with different colors) on im1 and view the resulting segmentation")

# Will crash if not in a "real" Qt loop
Gui.execLoop()

