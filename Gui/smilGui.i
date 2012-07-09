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
%module(directors="1") smilGuiPython
#endif // SWIGPYTHON

#ifdef SWIGJAVA
%module(directors="0") smilGuiJava
#endif // SWIGJAVA

#ifdef SWIGOCTAVE
%module(directors="1") smilGuiOctave
#endif // SWIGOCTAVE

#ifdef SWIGRUBY
%module(directors="1") smilGuiRuby
#endif // SWIGRUBY


%include smilCommon.i

%{
/* Includes the header in the wrapper code */
#include "DBaseObject.h"
#include "DImageViewer.hpp"

%}


//%include "DGui.h"
//%include "DCore.h"
%include "DBaseObject.h"
%include "DBaseImageViewer.h"
%include "DImageViewer.hpp"

// generate directors for Signal and Slot (for virtual methods overriding)
%feature("director") imageViewer;

TEMPLATE_WRAP_CLASS(imageViewer);


#ifdef USE_QT

%{
#include "DQtImageViewer.hpp"
#include "DQtImageViewer.hxx"

%}

%include "DQtImageViewer.hpp"
%include "DQtImageViewer.hxx"

TEMPLATE_WRAP_CLASS(qtImageViewer);

#ifdef SWIGPYTHON
%pythoncode %{

if ('qApp' in locals())==0:
  import sys
  try:
    from PyQt4 import QtGui
    qApp = QtGui.QApplication(sys.argv)
  except:
    qApp = None

%}
#endif // SWIGPYTHON

#endif // USE_QT


