import smilPython as sp
import time

# define which structuring element to use
se = sp.SquSE()

# get an image
imIn = sp.Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
# declare output image
imOut = sp.Image(imIn)

# Display input, temporary and output images
imIn.show("Input image")
imOut.show()

input("Hit the enter key to begin")
for i in range(0,8):
  s = "Open SE({:})".format(i)
  print(s)
  imOut.setName(s)

  r = sp.open(imIn,  imOut, se(i))

  # process GUI events and update the image display
  sp.Gui.processEvents()
  # an optional wait of 0.4 seconds
  time.sleep(0.4)

input("Hit the enter key to exit")

