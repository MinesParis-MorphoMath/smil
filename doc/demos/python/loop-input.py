import smilPython as sp

# define which structuring element to use
se = sp.SquSE()

# get an image
imIn = sp.Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
# declare output image
imOut = sp.Image(imIn)

# Display input, temporary and output images
imIn.show("Input image")
imOut.show()


for i in range(0, 8):
    s = "Open SE({:})".format(i)
    print(s)
    imOut.setName(s)

    r = sp.open(imIn, imOut, se(i))
    # save temporari result, if wanted
    r = sp.write(imOut, "res-tmp-{:03d}".format(i))
