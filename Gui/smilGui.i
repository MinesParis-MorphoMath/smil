// Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OR
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


%include smilCommon.i

SMIL_MODULE(smilGui)




%{
/* Includes the header in the wrapper code */
#include "Gui/include/DGui.h"
#include "Core/include/DImage.h"
#include "Gui/include/private/DImageViewer.hpp"
%}

%import smilCore.i


//////////////////////////////////////////////////////////
// Gui Instance
//////////////////////////////////////////////////////////

%include "Core/include/private/DInstance.hpp"
%template(GuiInstance) smil::UniqueInstance<smil::Gui>;
%include "include/DGuiInstance.h"

%include "include/DGui.h"

// generate directors for virtual methods (except those returning const char ptr)
%feature("director") baseImageViewer;
%feature("nodirector") baseImageViewer::getInfoString;
%feature("nodirector") baseImageViewer::getClassName;
%feature("nodirector") baseImageViewer::getName;

%include "include/DBaseImageViewer.h"
%include "include/private/DImageViewer.hpp"

TEMPLATE_WRAP_CLASS(smil::ImageViewer, ImageViewer);

#ifdef USE_QT

%{
#include "Qt/DQtImageViewer.hpp"
#include "Qt/DQtImageViewer.hxx"
#include "Qt/DQtGuiInstance.h"
%}

%include "Qt/DQtImageViewer.hpp"
TEMPLATE_WRAP_CLASS(QtImageViewer, QtImageViewer);

#ifdef SWIGPYTHON
%init
%{
  // Process Qt events from the Python input hook.
  PyOS_InputHook = qtLoop;
%}

%pythoncode %{

import os

def QtGui():
  envVars = ['JUPYTER_NOTEBOOKS', 'DisableQtGui', 'SMIL_DISABLE_GUI']
  for k in envVars:
    if k in os.environ:
      return not os.environ[k].lower() in ['yes', 'true', '1']
  return True


if QtGui():
  try:
    __IPYTHON__
    import IPython
    if IPython.version_info[0] >= 5:
      get_ipython().magic("%gui qt")
  except NameError:
    # print("err")
    pass

%}

#endif // SWIGPYTHON

#endif // USE_QT



#ifdef USE_AALIB
%{
#include "AALib/DAAImageViewer.hpp"
%}

%include "AALib/DAAImageViewer.hpp"
TEMPLATE_WRAP_CLASS(AaImageViewer, AaImageViewer);

#endif // USE_AALIB


