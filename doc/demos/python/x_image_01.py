# create an empty image
im = sp.Image(32, 32)
# draw a rectangle on it
for i in range(8, 24):
    for j in range(8, 24):
        im.setPixel(i, j, 255)
