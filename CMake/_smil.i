// Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
// Copyright (c) 2017-2024, Centre de Morphologie Mathematique
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.


%include smilCommon.i

SMIL_MODULE(smil_)

%feature("autodoc", "1");

#ifndef SWIGJAVA
%init
%{
  std::cout << "SMIL (Simple Morphological Image Library) ${SMIL_VERSION}" <<
      std::endl;
  std::cout << "Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES" <<
      std::endl;
  std::cout <<
      "Copyright (c) 2017-2024, CMM - Centre de Morphologie Mathematique" <<
      std::endl;
  std::cout << "All rights reserved." << std::endl;
      std::cout << std::endl;
%}
#endif // SWIGJAVA


// CMake generated list of interface files

${SWIG_INCLUDE_DEFINITIONS}



//
// #####    #   #   #####  #    #   ####   #    #
// #    #    # #      #    #    #  #    #  ##   #
// #    #     #       #    ######  #    #  # #  #
// #####      #       #    #    #  #    #  #  # #
// #          #       #    #    #  #    #  #   ##
// #          #       #    #    #   ####   #    #
//
#ifdef SWIGPYTHON

%pythoncode %{

import sys, gc, os
import time

if sys.version_info >= (3, 0, 0):
  import builtins as __builtin__
else:
  import __builtin__

from smilCorePython import *
from smilBasePython import *
from smilIOPython import *
#from smilMorphoPython import *

numpyOK = False
try:
  import numpy as np
except:
  numpyOK = True

__builtin__.dataTypes = [ ${DATA_TYPES_QUOTE_STR}, ]
__builtin__.imageTypes = [ ${IMAGE_TYPES_STR}, ]
__builtin__.imageTypesStr = ['Image_'+x for x in dataTypes]

#__builtin__.dataTypes = ["UINT8", "UINT16", "UINT32", "RGB",]
#__builtin__.imageTypes = [Image_UINT8, Image_UINT16, Image_UINT32, Image_RGB, ]

# ----------------------------------------------------------------------
#
def AboutSmil():
  print("SMIL (Simple Morphological Image Library) ${SMIL_VERSION}")
  print("Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES")
  print("Copyright (c) 2017-2024, CMM - Centre de Morphologie Mathematique")
  print("All rights reserved.")

def Version(verbose=False):
  version = "${SMIL_VERSION}"
  if not verbose:
    print(f"{version:}")
  else:
    print("SMIL (Simple Morphological Image Library) ${SMIL_VERSION}")
    #stype = ' '.join(__builtin__.dataTypes)
    stype = "${DATA_TYPES_STR}"
    print(f"Image Data types : {stype:}")

# ======================================================================
#
#     #####   #####      #    #    #    ##     #####  ######
#     #    #  #    #     #    #    #   #  #      #    #
#     #    #  #    #     #    #    #  #    #     #    #####
#     #####   #####      #    #    #  ######     #    #
#     #       #   #      #     #  #   #    #     #    #
#     #       #    #     #      ##    #    #     #    ######
#
# ----------------------------------------------------------------------
#
def _find_object_name(obj):
  names = []
  for referrer in gc.get_referrers(obj):
    if isinstance(referrer, dict):
      for k, v in referrer.items():
        if v is obj:
          names.append(k)
  if len(names) != 0:
    return names[-1]
  else:
    return ""


__builtin__._find_object_name = _find_object_name


# ----------------------------------------------------------------------
# this function works only inside this module because globals returns
# only objects defined inside this module
def _find_images(gbl_dict=None):
  if not gbl_dict:
    gbl_dict = globals()
  imgs = dict()
  for it in gbl_dict.items():
    for t in imageTypes:
      if isinstance(it[1], t):
        imgs[it[1]] = it[0]
  return imgs


__builtin__.getImages = _find_images


# ======================================================================
#
#     #    #    #     #     #####
#     #    ##   #     #       #
#     #    # #  #     #       #
#     #    #  # #     #       #
#     #    #   ##     #       #
#     #    #    #     #       #
#
# ----------------------------------------------------------------------
# doesn't work - see _find_images
def guess_images_name(gbl_dict=None):
  imgs = _find_images(gbl_dict)
  for im in imgs.keys():
    if im.getName() == '':
      im.setName(imgs[im])


# ----------------------------------------------------------------------
#
class showImageSlot(BaseImageEventSlot):
  def __init__(self):
    BaseImageEventSlot.__init__(self)

  def run(self, event):
    guess_images_name()


_showImageSlot = showImageSlot()


# ----------------------------------------------------------------------
#
class createImageSlot(BaseImageEventSlot):
  def run(self, event):
    img = event.sender
    img.onShow.connect(_showImageSlot)


# ----------------------------------------------------------------------
#
class deleteImageSlot(BaseImageEventSlot):
  def run(self, event):
    img = event.sender
    guess_images_name()


# ----------------------------------------------------------------------
#
core = Core.getInstance()
_newImageSlot = createImageSlot()
_delImageSlot = deleteImageSlot()
core.onBaseImageCreated.connect(_newImageSlot)
core.onBaseImageDestroyed.connect(_delImageSlot)


# ======================================================================
#
#     #    #    #    ##     ####   ######
#     #    ##  ##   #  #   #    #  #
#     #    # ## #  #    #  #       #####
#     #    #    #  ######  #  ###  #
#     #    #    #  #    #  #    #  #
#     #    #    #  #    #   ####   ######
#
# ----------------------------------------------------------------------
#
# doesn't work - see _find_images
def showAll():
  imgs = _find_images()
  for im in imgs.keys():
    im.show()


# ----------------------------------------------------------------------
# doesn't work - see _find_images
def hideAll():
  imgs = _find_images()
  for im in imgs.keys():
    im.hide()


# ----------------------------------------------------------------------
# doesn't work - see _find_images
def deleteAll():
  imgs = _find_images()
  for im in imgs.keys():
    im.hide()
    globals().pop(imgs[im], None)
    del im


# ----------------------------------------------------------------------
#
def autoCastBaseImage(baseImg):
  if not baseImg:
    return None
  typeStr = baseImg.getTypeAsString()
  if typeStr in dataTypes:
    imType = imageTypes[dataTypes.index(typeStr)]
    # Steal baseImg identity (kind of trick for python cast)
    return imType(baseImg, True)
  else:
    return None


# ----------------------------------------------------------------------
#
def Image(*args, **kargs):
  """
    Image()
    Image(shape=(256,256,1), type=${DEFAULT_IMAGE_TYPE})
    Image(szx[, szy, [szz]], type=${DEFAULT_IMAGE_TYPE})
    Image(im, clone=False, type=None)
    Image(ndarray)
    Image(path, raw=False)
    Image(path, raw=True, shape=(256,256,1), type=${DEFAULT_IMAGE_TYPE})
    Image(url)
    Image([path, path, ...], raw=False)
    Image([path, path, ...], raw=True, shape=(256,256,1), type=${DEFAULT_IMAGE_TYPE})

    * Image():
      Create an empty ${DEFAULT_IMAGE_TYPE} image.
    * Image(width, height [, depth]):
      Create a ${DEFAULT_IMAGE_TYPE} image with size 'width'x'height'[x'depth'].
    * Image(im):
      Create an image with same type and same size as 'im'.
    * Image(im, width, height [, depth]):
      Create an image with same type 'im' and with size
        'width x height [x depth]'.
    * Image("TYPE"):
      Create an empty image with the desired type.
      The available image types are: ${DATA_TYPES_STR}
    * Image("TYPE", width, height [, depth]):
      Will create an image with the desired type and dimensions.
    * Image(im, "TYPE"):
      Create an image with type 'TYPE' and with same size as 'im'.
    * Image("fileName"):
      Create an image and load the file "fileName".
      OBS : "filename" can be an URL if Smil was compiled and
      linked against curl library.
    * Image("fileName", "TYPE"):
      Create an image with type 'TYPE' and load the file "fileName".

    """

  paramTypes = {
      'shape' : list,
      'imtype' : str,
      'itype' : type,
      'clone': bool,
      'raw' : bool,
  }
  paramValues = {
      'shape' : [],
      'imtype' : dataTypes,
      'itype' : imageTypes,
      'clone' : [True, False],
      'raw' : [True, False],
  }

  #
  def validArgs(kargs):
    ok = True
    for k in kargs:
      if not k in paramTypes:
        print('ERROR : bad parameter {:s}'.format(k))
        ok = False
        continue
      if not isinstance(kargs[k], paramTypes[k]):
        print('ERROR : bad parameter type of {:s} : {:} (should be {:}'.
              format(k, type(kargs[k]), paramTypes[k]))
        ok = False
        continue
      if not kargs[k] in paramValues[k] and not isinstance(kargs[k], list):
        print('ERROR : bad parameter value : {:s} {:}'.format(k, kargs[k]))
        print('        Valid values : {:}'.format(paramValues[k]))
        ok = False
    return ok

  #
  def _fillImage(img):
    if img.isAllocated():
      try:
        fillValue = type(img).getDataTypeMin()
        fill(img, fillValue)
      except:
        pass
    return img



  #
  def dictKeyByValue(d, value):
    for k, v in d.items():
      if value == v:
        return k
    return None

  #
  def dType2index(dt = 'UINT8'):
    if dt in dataTypes:
      return dataTypes.index(dt)
    return 0

  #
  def dType2dFunc(dt = 'UINT8'):
    if dt in dFuncByStr.keys():
      return dFuncByStr[dt]

    print('ERROR : bad image type : {:}'.format(dt))
    print('        Valid types : {:}'.format(paramValues['imtype']))
    return None

  #
  def getFirstInts(args):
    out = []
    for i in range(0, len(args)):
      if isinstance(args[i], int):
        out.append(args[i])
      else:
        break
      if len(out) >= 3:
        break
    return out



  #
  # M A I N
  #
  if not validArgs(kargs):
    return None

  argNbr = len(args)
  argTypeStr = [str(type(a)) for a in args]

  #
  #
  dFuncByStr = {dataTypes[i]:imageTypes[i] for i in range(0, len(dataTypes))}
  dFuncByType = {('Image_' + dataTypes[i]):imageTypes[i] for i in range(0, len(dataTypes))}

  img = None
  fillImg = False
  imFunc = None

  #
  # no args or just 'imtype=xxx'
  if len(args) == 0:
    shape = [256, 256, 1]
    if 'imtype' in kargs:
      imFunc = dType2dFunc(kargs['imtype'])
    else:
      imFunc = imageTypes[0]
    if 'shape' in kargs:
      shape = kargs['shape']
      while len(shape) < 3:
        shape.append(1)
    if not imFunc is None:
      img = imFunc(shape[0], shape[1], shape[2])
      _fillImage(img)
    return img

  #
  # first args are image dimensions and optional 'imtype'
  if isinstance(args[0], int):
    imFunc = None
    dims = getFirstInts(args)
    nInt = len(dims)
    while len(dims) < 3:
      dims.append(1)
    if 'imtype' in kargs:
      imFunc = dType2dFunc(kargs['imtype'])
    else:
      if len(args) > nInt and isinstance(args[nInt], str):
        imFunc = dType2dFunc(args[nInt])
      else:
        imFunc = imageTypes[0]

    if not imFunc is None:
      img = imFunc(dims[0], dims[1], dims[2])
      _fillImage(img)
    return img

  #
  # first arg is image type ('UINTnn' | 'RGB')
  if isinstance(args[0], str) and args[0] in dataTypes:
    imFunc = dType2dFunc(args[0])
    if not imFunc is None:
      shape = [256, 256, 1]
      if 'shape' in kargs:
        shape = kargs['shape']
        while len(shape) < 3:
          shape.append(1)
      img = imFunc(shape[0], shape[1], shape[2])
      #img = imgFunc(*args[1:])
      _fillImage(img)
      return img

  #
  # Image('nomdefichier.ext' | 'http://')
  if isinstance(args[0], str):
    # Create/load from an existing image fileName

    if 'imtype' in kargs:
      imFunc = dType2dFunc(kargs['imtype'])
    if imFunc is None:
      if len(args) > 1 and isinstance(args[1], str):
        if args[1] in dataTypes:
          imFunc = dType2dFunc(args[1])

    urlPrefix = ('http://', 'https://')
    if (os.path.exists(args[0]) or args[0].startswith(urlPrefix)):
      if not imFunc is None:
        img = imFunc()
        read(args[0], img)
      else:
        baseImg = createFromFile(args[0])
        if baseImg != None:
          img = autoCastBaseImage(baseImg)
      return img
    else:
      print("File or URL not found: ", args[0])
      return img

  #
  # Image(imIn) - First arg is an image
  if type(args[0]).__name__ in dFuncByType:
    srcType = type(args[0]).__name__
    srcIm = args[0]

    if 'imtype' in kargs:
      imFunc = dType2dFunc(kargs['imtype'])
    if imFunc is None:
      if len(args) > 1 and isinstance(args[1], str):
        if args[1] in dataTypes:
          imFunc = dType2dFunc(args[1])

    clone = False
    if 'clone' in kargs:
      clone = kargs['clone']

    if imFunc is None:
      if srcType in dFuncByType:
        imFunc = dFuncByType[srcType]
        img = imFunc(srcIm, clone)
        if not clone:
          _fillImage(img)
    else:
      img = imFunc()
      img.setSize(srcIm)
      if clone:
        copy(srcIm, img)
      else:
        _fillImage(img)
    return img

  #
  # Image(arr) - numpy array
  # if 'numpy.ndarray' in str(type(args[0])):
  if type(args[0]).__name__ == 'ndarray':
    img = Image()
    img.fromNumpyArray(args[0])
    return img
#    return NumpyInt(*args)

  # Let C++ try to solve it
  img = imageTypes[0](*args)
  fillImg = True
  _fillImage(img)

  return img


# ----------------------------------------------------------------------
#
def Images(nbr, *args, **keywords):
  return [Image(*args) for i in range(nbr)]


# ======================================================================
#
#     #          #    #    #  #    #
#     #          #    ##   #  #   #
#     #          #    # #  #  ####
#     #          #    #  # #  #  #
#     #          #    #   ##  #   #
#     ######     #    #    #  #    #
#
# ----------------------------------------------------------------------
#
class linkManager:
  def __init__(self):
    self.links = []

  class _linkArgs(list):
    def __init__(self, link):
      list.__init__(self)
      self._link = link

    def __setitem__(self, num, val):
      prevVal = list.__getitem__(self, num)
      list.__setitem__(self, num, val)
      if not self._link.run(None):
        list.__setitem__(self, num, prevVal)

  class link(BaseImageEventSlot):
    def __init__(self, imWatch, func, *args):
      BaseImageEventSlot.__init__(self)
      self.imWatch = imWatch
      self.func = func
      self.args = linkManager._linkArgs(self)
      for a in args:
        self.args.append(a)
      self.verified = False
      if self.run(None):
        self.verified = True
        self.imWatch.onModified.connect(self)

    def __del__(self):
      self.imWatch.onModified.disconnect(self)

    def run(self, event):
      try:
        for obj in self.args:
          if hasattr(obj, "getClassName"):
            if obj.getClassName() == "Image":
              obj.setSize(self.imWatch)
        self.func(*self.args)
        return True
      except Exception as e:
        print("Link function error:\n")
        print(e)
        return False

    def __str__(self):
      res = _find_object_name(self.imWatch) + " -> "
      res += self.func.__name__ + " "
      for obj in self.args:
        if hasattr(obj, "getClassName"):
          oName = _find_object_name(obj)
          if oName != "":
            res += oName + " "
          else:
            res += str(obj) + " "
        else:
          res += str(obj) + " "
      return res

  def __getitem__(self, num):
    return self.links[num]

  def __setitem__(self, num, l):
    self.links[num] = l

  def find(self, imWatch, func=None, *args):
    res = []
    for l in self.links:
      if l.imWatch == imWatch:
        if func == None or l.func == func:
          if args == () or l.args == args:
            res.append(l)
    return res

  def add(self, imWatch, func, *args):
    if self.find(imWatch, func, *args):
      print("link already exists.")
      return
    l = self.link(imWatch, func, *args)
    if l.verified:
      self.links.append(l)

  def remove(self, imWatch, func=None, *args):
    if type(imWatch) == int:  # remove Nth link
      self.links.remove(self.links[imWatch])
      return
    _links = self.find(imWatch, func, *args)
    if _links:
      for l in _links:
        self.links.remove(l)
      return
    self.links.append(self.link(imWatch, func, *args))

  def list(self):
    i = 0
    for l in self.links:
      print("#" + str(i), l)
      i += 1

  def clear(self):
    for l in self.links:
      del l.args
      del l
    self.links = []

  def __del__(self):
    self.clear()


# ======================================================================
#
#     #####   ######  #    #   ####   #    #
#     #    #  #       ##   #  #    #  #    #
#     #####   #####   # #  #  #       ######
#     #    #  #       #  # #  #       #    #
#     #    #  #       #   ##  #    #  #    #
#     #####   ######  #    #   ####   #    #
#
# ----------------------------------------------------------------------
# Benchmarking Smil tests (need to be compiled with TEST option set to ON)
#
def bench(func, *args, **keywords):
  """
    bench(function, [func_args], [options]):
      Execute bench. Return the mean execution time (in msecs) for one
      function execution.
    Available options:
      * nbr_runs: number of times the function will be executed (default is 1E3)
      * print_res: print results (default is True)
    """
  #default values
  nbr_runs = 1E3
  print_res = True
  add_str = None

  im_size = None
  im_type = None
  se_type = None

  if "nbr_runs" in keywords: nbr_runs = keywords["nbr_runs"]
  if "print_res" in keywords: print_res = keywords["print_res"]
  if "add_str" in keywords: add_str = keywords["add_str"]

  for arg in args:
    if not im_size:
      if type(arg) in imageTypes:
        im_size = arg.getSize()
        im_type = arg.getTypeAsString()
    if not se_type:
      if hasattr(arg, "getClassName") and hasattr(arg, "homothety"):
        se_type = arg.getClassName()

  t1 = time.time()

  for i in range(int(nbr_runs)):
    func(*args)

  t2 = time.time()

  retval = (t2 - t1) * 1E3 / nbr_runs

  if print_res:
    buf = func.__name__ + "\t"
    if im_size or add_str or se_type:
      buf += "("
    if im_size:
      buf += im_type + " " + str(im_size[0])
      if im_size[1] > 1: buf += "x" + str(im_size[1])
      if im_size[2] > 1: buf += "x" + str(im_size[2])
    if add_str:
      buf += " " + add_str
    if se_type:
      buf += " " + se_type
    if im_size or add_str or se_type:
      buf += ")"
    buf += ":\t" + "%.2f" % retval + " msecs"
    print(buf)

  return retval

#
#  ####   #       #####
# #    #  #       #    #
# #    #  #       #    #
# #    #  #       #    #
# #    #  #       #    #
#  ####   ######  #####
#

def oldImage(*args):
    """
    * Image():
      Create an empty ${DEFAULT_IMAGE_TYPE} image.
    * Image(width, height [, depth]):
      Create a ${DEFAULT_IMAGE_TYPE} image with size 'width'x'height'[x'depth'].
    * Image(im):
      Create an image with same type and same size as 'im'.
    * Image(im, width, height [, depth]):
      Create an image with same type 'im' and with size
        'width x height [x depth]'.
    * Image("TYPE"):
      Create an empty image with the desired type.
      The available image types are: ${DATA_TYPES_STR}
    * Image("TYPE", width, height [, depth]):
      Will create an image with the desired type and dimensions.
    * Image(im, "TYPE"):
      Create an image with type 'TYPE' and with same size as 'im'.
    * Image("fileName"):
      Create an image and load the file "fileName".
      OBS : "filename" can be an URL if Smil was compiled and
      linked against curl library.
    * Image("fileName", "TYPE"):
      Create an image with type 'TYPE' and load the file "fileName".

    """

    argNbr = len(args)
    argTypeStr = [ str(type(a)) for a in args ]

    img = None
    fillImg = False

    if argNbr==0:
        # No argument -> return default image type
        img = imageTypes[0](256,256)
        fillImg = True

    elif type(args[0])==int:
        # First arg is a number (should be a size)
        img = imageTypes[0](*args)
        fillImg = True

    elif 'numpy.ndarray' in str(type(args[0])):
        return NumpyInt(*args)

    elif type(args[0]) in imageTypes or hasattr(args[0], "getTypeAsString"):
        # First arg is an image
        srcIm = args[0]
        if type(srcIm) in imageTypes:
            srcImgType = type(srcIm)
        else:
            srcImgType = imageTypes[dataTypes.index(srcIm.getTypeAsString())]
        if argNbr>1:
          if type(args[1])==type(""):
              # Second arg is an image type string ("UINT8", ...)
              if args[1] in dataTypes:
                  imgType = imageTypes[dataTypes.index(args[1])]
                  img = imgType()
                  img.setSize(srcIm)
              else:
                  print("Unknown image type: " + args[1])
                  print("List of available image types: " +  ", ".join(dataTypes))
          else:
              img = srcImgType(*args[1:])
        else:
            img = srcImgType(srcIm, False) # (don t clone data)
        fillImg = True

    elif args[0] in dataTypes:
        # First arg is an image type string ("UINT8", ...)
        imgType = imageTypes[dataTypes.index(args[0])]
        img = imgType(*args[1:])
        fillImg = True

    elif argNbr > 0 and type(args[0]) == str:
        # Create/load from an existing image fileName
        urlPrefix = ('http://', 'https://')
        if (os.path.exists(args[0]) or args[0].startswith(urlPrefix)):
            if argNbr>1 and args[1] in dataTypes:
                imgType = imageTypes[dataTypes.index(args[1])]
                img = imgType()
                read(args[0], img)
            else:
                baseImg = createFromFile(args[0])
                if baseImg != None:
                  img = autoCastBaseImage(baseImg)
        else:
            print("File not found: ", args[0])

    else:
        img = imageTypes[0](*args)
        fillImg = True

    if fillImg and img.isAllocated():
      try:
        fillValue = type(img).getDataTypeMin()
        fill(img, fillValue)
      except:
        pass
    return img

%}

#endif // SWIGPYTHON
