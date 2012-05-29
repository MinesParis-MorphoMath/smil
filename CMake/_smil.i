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
//     * Neither the name of the University of California, Berkeley nor the
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
%}

// CMake generated list of interface files

${SWIG_INCLUDE_DEFINITIONS}




#ifdef SWIGPYTHON

%pythoncode %{

import sys, gc
import time, new

${PYTHON_IMPORT_MODULES}

${SWIG_IMAGE_TYPES}


def Image(*args):
    argNbr = len(args)
    argTypeStr = [ str(type(a)) for a in args ]
    
    img = 0
    if argNbr==0:
	img = imageTypes[0]()
    elif argNbr>=2:
	img = imageTypes[0](*args)
    else:
	if argTypeStr[0].rfind("Image_")!=-1:
	  srcIm = args[0]
	  if argNbr==1:
	    img = createImage(srcIm)
	else:
	    img = imageTypes[args[1]](srcIm.getWidth(), srcIm.getHeight(), srcIm.getDepth())
    # img.show = new.instancemethod(show_with_name, img, img.__class__)
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
	name = find_object_names(img)[1]
	img.setName(name)
    img.c_show(name, labelImage)

def showLabel_with_name(img, name=None):
    if not name:
	name = find_object_names(img)[1]
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

print "SMIL (Simple Morphological Image Library)"
print "Copyright (c) 2011, Matthieu FAESSEL and ARMINES"
print "All rights reserved."
print ""

%}

#endif // SWIGPYTHON
