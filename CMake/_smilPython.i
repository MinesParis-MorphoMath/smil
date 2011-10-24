%module smilPython


%feature("autodoc", "1");


/*%include <windows.i> */
%include <std_string.i>
%include <typemaps.i>

%rename(__lshift__)  operator<<; 


// CMake generated wrap macros

${SWIG_TEMPLATE_WRAP_DEFINITIONS}


// CMake generated list of interface files

${SWIG_INCLUDE_DEFINITIONS}


TEMPLATE_WRAP_CLASS(Image);


%pythoncode %{

from PyQt4 import QtGui, QtCore
import gc, sys, time
import new

if ('qtApp' in locals())==0:
  _qtApp = QtGui.QApplication(sys.argv)


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