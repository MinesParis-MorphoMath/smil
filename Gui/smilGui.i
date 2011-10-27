%module smilPython

%{
/* Includes the header in the wrapper code */
#include "DImage.hpp"
#include "DGui.h"
#include "DImage.hxx"

#ifdef USE_QT
#include "Qt/QtApp.h"
#endif // USE_QT
%}


%include "DGui.h"
#ifdef USE_QT
%include "Qt/QtApp.h"
#endif // USE_QT


#ifdef SWIGPYTHON
%pythoncode %{

from PyQt4 import QtGui, QtCore
import sys

if ('qtApp' in locals())==0:
  _qtApp = QtGui.QApplication(sys.argv)

%}
#endif // SWIGPYTHON
