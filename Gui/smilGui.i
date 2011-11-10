%module smilPython

%{
/* Includes the header in the wrapper code */
#include "DImage.hpp"
#include "DGui.h"
#include "DImage.hxx"

%}


%include "DGui.h"


#ifdef SWIGPYTHON
%pythoncode %{

from PyQt4 import QtGui, QtCore
import sys

if ('qtApp' in locals())==0:
  _qtApp = QtGui.QApplication(sys.argv)

%}
#endif // SWIGPYTHON
