 
from smilPython import *
import time

sx = 1024
sy = 1024
bench_nruns = 1E3


seTypes = "hSE, sSE"

def bench(func, *args, **keywords):
    #default values
    nbr_runs = 1E3
    print_res = True
    add_str = None
    
    im_size = None
    im_type = None
    se_type = None
    
    if keywords.has_key("nbr_runs"): nbr_runs = keywords["nbr_runs"]
    if keywords.has_key("print_res"): print_res = keywords["print_res"]
    if keywords.has_key("add_str"): add_str = keywords["add_str"]
    
    for arg in args:
      if not im_size:
	if type(arg) in imageTypes:
	  im_size = arg.getSize()
	  im_type = arg.getTypeAsString()
      if not se_type:
	if hasattr(arg, "__module__") and arg.__module__ == "smilMorphoPython":
	  arg_ts = str(type(arg)).split(".")[1][:-2]
	  if arg_ts in seTypes:
	    se_type = arg_ts
	    
    t1 = time.time()
    
    for i in range(int(nbr_runs)):
      func(*args)

    t2 = time.time()

    retval = (t2-t1)*1E3/nbr_runs
    
    buf = func.func_name + "\t"
    if im_size or add_str or se_type:
      buf += "("
    if im_size:
      buf += im_type + " " + str(im_size[0])
      if im_size[1]>1: buf += "x" + str(im_size[1])
      if im_size[2]>1: buf += "x" + str(im_size[2])
    if add_str:
      buf += " " + add_str
    if se_type:
      buf += " " + se_type
    if im_size or add_str or se_type:
      buf += ")"
    buf += ":\t" + "%.2f" % retval + " msecs"
    print buf
    return retval

# Load an image
imIn = Image("http://cmm.ensmp.fr/~faessel/smil/images/DNA_small.png")
#imIn.show()

for imType in (Image_UINT8,):
  tmpIm = imType(imIn)
  im1 = imType(sx, sy)
  im2 = imType(im1)
  im3 = imType(im1)
  
  copy(imIn, tmpIm)
  resize(tmpIm, im1)

  bench(fill,im1, 0)
  bench(inv, im1, im2)
  bench(sup, im1, im2, im3)

  bench(dilate, im1, im2, hSE(1))
  bench(dilate, im1, im2, sSE(1))
  bench(erode, im1, im2, hSE(1))
  bench(erode, im1, im2, sSE(1))
  bench(open, im1, im2, hSE(1))
  bench(open, im1, im2, sSE(1))
  bench(close	, im1, im2, hSE(1))
  bench(close	, im1, im2, sSE(1))

  print 