#!/usr/bin/python

import smilPython as smil
from PIL import Image
import numpy as np


def ImageFromArray(im_arr):
    im = smil.Image(*im_arr.shape[::-1])
    im_np = im.getNumpyArray()
    im_np[:] = im_arr.transpose()[:]
    return im


in_file = "init.png"
Size = 20
tolerance = 2
step = 10
NN = 100

img = Image.open(in_file)
XX, YY = img.size
img_out = np.zeros((YY, XX), np.uint8)

for yy in np.arange(0, YY, NN):
    for xx in np.arange(0, XX, NN):
        img_ii = np.array(img)[yy : min(yy + NN, YY), xx : min(xx + NN, XX)].transpose()
        im_in = smil.Image(*img_ii.shape)
        im_in_np = im_in.getNumpyArray()
        im_in_np[:] = img_ii[:]

        im_out = smil.Image(im_in)
        smil.parsimoniousPathOpening(im_in, Size, tolerance, step, False, im_out)
        im_out_np = im_out.getNumpyArray().transpose()
        img_out_slice = img_out[yy : min(yy + NN, YY), xx : min(xx + NN, XX)]
        img_out_slice[:] = im_out_np[:]

img_reconstr = ImageFromArray(img_out)
img_out = ImageFromArray(img_out)
im_in = ImageFromArray(np.array(img))

smil.geoBuild(img_out, im_in, img_reconstr)
smil.write(img_out, "init_path_open.png")
smil.write(img_reconstr, "init_path_open_reconstr.png")
