%module smilPython



%feature("autodoc", "1");

/*%include <windows.i> */
%include <std_string.i>
%include <typemaps.i>

%rename(__lshift__)  operator<<; 

%define TEMPLATE_WRAP_CLASS(_class) 
  %template(_class ## _UINT8) _class<UINT8>;
  %template(_class ## _UINT16) _class<UINT16>;
/*  %template(_class ## _UINT32) _class<UINT32>;*/
%enddef

%define TEMPLATE_WRAP_FUNC(func)
  %template(func) func<UINT8>;
/*  %template(func) func<UINT16>; */
/*  %template(func) func<UINT32>; */
%enddef

%define TEMPLATE_WRAP_FUNC2(func)
  %template(func) func<UINT8,UINT8>;
  %template(func) func<UINT8,UINT16>;
  %template(func) func<UINT16,UINT8>;
  %template(func) func<UINT32>;
%enddef

%define TEMPLATE_WRAP_FUNC_IMG(func) 
  %template(func) func<Image_UINT8>;
  %template(func) func<Image_UINT16>;
  %template(func) func<Image_UINT32>;
%enddef


${SWIG_INCLUDE_DEFINITIONS}

TEMPLATE_WRAP_CLASS(Image);


%pythoncode %{

from PyQt4 import QtGui, QtCore
import gc, sys, time
import new

if ('qtApp' in locals())==0:
  _qtApp = QtGui.QApplication(sys.argv)


def find_names(obj):
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

def show_with_name(img):
    name = find_names(img)[1]
    img._show(name)

imageTypes = ( Image_UINT8, Image_UINT16 )

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
    # la classe python...
    img.show = new.instancemethod(show_with_name, img, img.__class__)
    return img

%}