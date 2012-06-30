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
%}


%init
%{
    #include <iostream>

    std::cout << "SMIL (Simple Morphological Image Library) ${SMIL_VERSION}" << std::endl;
    std::cout << "Copyright (c) 2011, Matthieu FAESSEL and ARMINES" << std::endl;
    std::cout << "All rights reserved." << std::endl;
    std::cout << std::endl;
%}


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
    * imageMb("TYPE"): create an empty image with the desired type.
      The available image types are: ${DATA_TYPES_STR}
    * Image("TYPE", width, height [, depth]): will create an image with the desired type and dimensions.
    * Image(im, "TYPE"): create an image with type 'TYPE' and with same type and same size than 'im'.
    * Image("fileName"): create an image and load the file "fileName".
    """

    argNbr = len(args)
    argTypeStr = [ str(type(a)) for a in args ]
    
    img = 0
    
    if argNbr==0: # No argument -> return default image type
	img = imageTypes[0]()
	
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
	    img = srcImgType(srcIm)
	    
    elif args[0] in dataTypes:
	imgType = imageTypes[dataTypes.index(args[0])]
	print args[1:]
	img = imgType(*args[1:])

    elif argNbr==1 and os.path.exists(args[0]):
	img = imageTypes[0]()
	read(args[0], img)
    
    else:
	img = imageTypes[0](*args)
	
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


%}

#endif // SWIGPYTHON
