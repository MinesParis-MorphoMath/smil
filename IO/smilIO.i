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


#ifdef SWIGPYTHON
%module smilIOPython
#endif // SWIGPYTHON

#ifdef SWIGJAVA
%module smilIOJava
#endif // SWIGJAVA

#ifdef SWIGOCTAVE
%module smilIOOctave
#endif // SWIGOCTAVE

#ifdef SWIGRUBY
%module smilIORuby
#endif // SWIGRUBY


%include smilCommon.i

%{
/* Includes the header in the wrapper code */
#include "Core/include/DImage.h"
#include "Core/include/private/DTypes.hpp"
#include "DIO.h"
#include "DImageIO_RAW.hpp"
%}
 

%include "DCommonIO.h"

%include "Core/include/private/DTypes.hpp"
%include "DIO.h"
%include "DImageIO.hpp"

%include "DImageIO_RAW.hpp"


// Import smilCore to have correct function signatures (arguments with Image_UINT8 instead of Image<unsigned char>)
%import smilCore.i

TEMPLATE_WRAP_FUNC(read);
TEMPLATE_WRAP_FUNC(write);

TEMPLATE_WRAP_SUPPL_FUNC(write);
TEMPLATE_WRAP_SUPPL_FUNC(read);


TEMPLATE_WRAP_FUNC(readRAW);
TEMPLATE_WRAP_FUNC(writeRAW);

TEMPLATE_WRAP_SUPPL_FUNC(readRAW);
TEMPLATE_WRAP_SUPPL_FUNC(writeRAW);
