// Smil
// Copyright (c) 2010 Matthieu Faessel
//
// This file is part of Smil.
//
// Smil is free software: you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// Smil is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with Smil.  If not, see
// <http://www.gnu.org/licenses/>.



#ifdef SWIGPYTHON
%module smilPython
#endif // SWIGPYTHON

#ifdef SWIGJAVA
%module smilJava
#endif // SWIGJAVA


%feature("autodoc", "1");


/*%include <windows.i> */
%include <std_string.i>
%include <typemaps.i>
//%include cpointer.i

%rename(__lshift__)  operator<<; 
%ignore *::operator=;

// CMake generated wrap macros

${SWIG_TEMPLATE_WRAP_DEFINITIONS}


// CMake generated list of interface files

${SWIG_INCLUDE_DEFINITIONS}


TEMPLATE_WRAP_CLASS(Image);

#ifdef SWIGPYTHON

%pythoncode %{

import sys, gc
import time, new

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

def show_with_name(img, name=None):
    if not name:
	name = find_object_names(img)[1]
    img.c_show(name)

for t in imageTypes:
    t.c_show = t.show
    t.show = show_with_name

%}

#endif // SWIGPYTHON

%feature("autodoc", "1");
