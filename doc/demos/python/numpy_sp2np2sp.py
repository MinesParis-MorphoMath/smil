import smilPython as sm

# read a 16 bits RAW Image
file = "Slices-16.raw"
im16 = sm.Image("UINT16")
sp.readRAW(file, 700, 700, 700, im16)

# Let's convert 8 bit input image

# get a pointer to a numpy array
p16 = im16.getNumpyArray()

# scale pixel values from 2**16 to 2**8
p16 //= 256

# get a new 8 bit numpy array
p8 = p.astype("uint8")

# create a 8 bits image with the same dimensions of the 16 bit image
im8 = sm.Image(im16, "UINT8")
# come back to Smil Image
im8.fromNumpyArray(p8)
