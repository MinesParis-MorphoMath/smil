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
