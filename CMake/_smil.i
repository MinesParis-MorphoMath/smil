// Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



#ifdef SWIGPYTHON
%module smilPython
#endif // SWIGPYTHON

#ifdef SWIGJAVA
%module smilJava
#endif // SWIGJAVA

#ifdef SWIGOCTAVE
%module smilOctave
#endif // SWIGOCTAVE


%feature("autodoc", "1");

%{
#include "stddef.h"
#include <iostream>
%}


#ifndef SWIGJAVA
%init
%{
    std::cout << "SMIL (Simple Morphological Image Library) ${SMIL_VERSION}" << std::endl;
    std::cout << "Copyright (c) 2011, Matthieu FAESSEL and ARMINES" << std::endl;
    std::cout << "All rights reserved." << std::endl;
    std::cout << std::endl;
%}
#endif // SWIGJAVA


// CMake generated list of interface files

${SWIG_INCLUDE_DEFINITIONS}




#ifdef SWIGPYTHON

%pythoncode %{

import sys, gc, os
import time, new

${PYTHON_IMPORT_MODULES}

dataTypes = [ ${DATA_TYPES_QUOTE_STR}, ]
imageTypes = [ ${IMAGE_TYPES_STR}, ]


def Image(*args):
    """
    * Image(): create an empty ${DEFAULT_IMAGE_TYPE} image.
    * Image(width, height [, depth]): create a ${DEFAULT_IMAGE_TYPE} image with size 'width'x'height'[x'depth'].
    * Image(im): create an image with same type and same size than 'im'.
    * Image("TYPE"): create an empty image with the desired type.
      The available image types are: ${DATA_TYPES_STR}
    * Image("TYPE", width, height [, depth]): will create an image with the desired type and dimensions.
    * Image(im, "TYPE"): create an image with type 'TYPE' and with same size than 'im'.
    * Image("fileName"): create an image and load the file "fileName".
    * Image("fileName", "TYPE"): create an image with type 'TYPE' and load the file "fileName".
    """

    argNbr = len(args)
    argTypeStr = [ str(type(a)) for a in args ]
    
    img = None
    
    if argNbr==0: # No argument -> return default image type
	img = imageTypes[0]()

    elif type(args[0])==int: # First arg is a number (should be a size)
	img = imageTypes[0](*args)
	fill(img, 0)
	
    elif type(args[0]) in imageTypes: # First arg is an image
	srcIm = args[0]
	srcImgType = type(args[0])
	if argNbr>1:
	  if type(args[1])==type(""):
	      if args[1] in dataTypes: # Second arg is an image type string ("UINT8", ...)
		  imgType = imageTypes[dataTypes.index(args[1])]
		  img = imgType()
		  img.setSize(srcIm)
	      else:
		  print "Unknown image type: " + args[1]
		  print "List of available image types: " +  ", ".join(dataTypes)
	  else:
	      img = srcImgType(*args[1:])
	else:
	    img = srcImgType(srcIm, False) # (don't clone data)
	fill(img, 0)
	    
    elif args[0] in dataTypes: # First arg is an image type string ("UINT8", ...)
	imgType = imageTypes[dataTypes.index(args[0])]
	img = imgType(*args[1:])
	fill(img, 0)

    # Create/load from an existing image fileName
    elif argNbr>0 and (os.path.exists(args[0]) or args[0][:7]=="http://"):
	if argNbr>1 and args[1] in dataTypes:
	    imgType = imageTypes[dataTypes.index(args[1])]
	    img = imgType()
	    read(args[0], img)
	else:
	    img = imageTypes[0](args[0])
    
    else:
	img = imageTypes[0](*args)
	fill(img, 0)
	
    return img



def find_object_names(obj):
    frame = sys._getframe()
    for frame in iter(lambda: frame.f_back, None):
	frame.f_locals
    result = []
    for referrer in gc.get_referrers(obj):
	if isinstance(referrer, dict):
	    for k, v in referrer.iteritems():
		if v is obj:
		    result.append(k)
    return result

def show_with_name(img, name=None, labelImage = False):
    if not name:
	name = find_object_names(img)[-1]
	img.setName(name)
    img.c_show(name, labelImage)

def showLabel_with_name(img, name=None):
    if not name:
	name = find_object_names(img)[-1]
	img.setName(name)
    img.c_showLabel(name)

def imInit(img, *args):
    img.__init0__(args)
    name = find_object_names(img)
    print name
    
for t in imageTypes:
    t.c_show = t.show
    t.show = show_with_name
    t.c_showLabel = t.showLabel
    t.showLabel = showLabel_with_name
#    t.__del0__ = t.__del__
#    t.__del__ = deleteImage


seTypes = "HexSE, SquSE"

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


    
class _linkManager():
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
	  
    class link(EventSlot):
      def __init__(self, imWatch, func, *args):
	EventSlot.__init__(self)
	self.imWatch = imWatch
	self.func = func
	self.args = _linkManager._linkArgs(self)
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
	      if obj.getClassName()=="Image":
		obj.setSize(self.imWatch)
	  self.func(*self.args)
	  return True
	except Exception, e:
	  print "Link function error:\n"
	  print e
	  return False
	
      def __str__(self):
	res = find_object_names(self.imWatch)[-1] + " -> "
	res += self.func.__name__ + " "
	for obj in self.args:
	  if hasattr(obj, "getClassName"):
	    res += find_object_names(obj)[-1] + " "
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
	if l.imWatch==imWatch:
	  if func==None or l.func==func:
	    if args==() or l.args==args:
	      res.append(l)
      return res
      
    def add(self, imWatch, func, *args):
      if self.find(imWatch, func, *args):
	print "link already exists."
	return
      l = self.link(imWatch, func, *args)
      if l.verified:
	self.links.append(l)
      
    def remove(self, imWatch, func=None, *args):
      if type(imWatch)==int: # remove Nth link
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
	print "#" + str(i), l
	i += 1
	
    def clear(self):
      for l in self.links:
	del l.args
	del l
      self.links = []
    
    def __del__(self):
      self.clear()
      
links = _linkManager()


%}

#endif // SWIGPYTHON
