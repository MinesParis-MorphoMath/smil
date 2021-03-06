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




%include smilCommon.i

SMIL_MODULE(smilVtk)


%{
/* Includes the header in the wrapper code */
#include "DVtkInterface.hpp"
#include "DVtkIO.hpp"
%}

%import smilCore.i


%include "DVtkInterface.hpp"

TEMPLATE_WRAP_CLASS(VtkInt,VtkInt)
TEMPLATE_WRAP_SUPPL_CLASS(VtkInt,VtkInt)


%include "DVtkIO.hpp"

TEMPLATE_WRAP_FUNC(readDICOM)
TEMPLATE_WRAP_SUPPL_FUNC(readDICOM)

#ifdef SWIGPYTHON

%pythoncode %{

def VtkInt(*args):
    """
    * Create a SharedImage interface with a vtkImageData
    """

    argNbr = len(args)
    argTypeStr = [ str(type(a)) for a in args ]
    
    if argNbr==0 or argTypeStr[0]!="<type 'vtkobject'>" or args[0].GetClassName()!='vtkImageData':
      print "You must specify a vtkImageData"
      return
    
    im = args[0]
    dt = im.GetScalarTypeAsString()
    
    if dt=="unsigned char":
      return VtkInt_UINT8(im)
    elif dt=="unsigned short":
      return VtkInt_UINT16(im)
    elif dt=="short":
      return VtkInt_INT16(im)
%}

        
#endif // SWIGPYTHON



