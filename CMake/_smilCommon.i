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


%feature("autodoc", "1");


//////////////////////////////////////////////////////////
// Common includes
//////////////////////////////////////////////////////////

%{
#include "DErrors.h"
#include "DTypes.hpp"
#include "DBaseObject.h"
#include "DBaseImage.h"
#include "DImage.hpp"
#include "DImage.hxx"
#include "DSharedImage.hpp"
#include "DInstance.hpp"
#include "DCoreInstance.h"
#include "DSlot.h"
#include "DSignal.h"
#include "DCoreEvents.h"
%}


//////////////////////////////////////////////////////////
// Module definitions
//////////////////////////////////////////////////////////

#ifdef SWIGPYTHON
%define SMIL_MODULE(libname)
    %module(directors="1") libname ## Python
%enddef
#endif // SWIGPYTHON

#ifdef SWIGJAVA
%define SMIL_MODULE(libname)
    %module(directors="0") libname ## Java
%enddef
// Problemes de directors avec Java... (a resoudre)
#endif // SWIGJAVA

#ifdef SWIGOCTAVE
%define SMIL_MODULE(libname)
    %module(directors="1") libname ## Octave
%enddef
#endif // SWIGOCTAVE

#ifdef SWIGRUBY
%define SMIL_MODULE(libname)
    %module(directors="1") libname ## Ruby
%enddef
#endif // SWIGRUBY


#define _DCORE
#define _DBASE
#define _DIO
#define _DGUI
#define _DMORPHO


%include cpointer.i
%include std_string.i
%include typemaps.i



%rename(__lshift__)  operator<<; 
%ignore *::operator=;


#ifdef SWIGJAVA
%ignore *::operator+;
%ignore *::operator+=;
%ignore *::operator-;
%ignore *::operator-=;
%ignore *::operator*;
%ignore *::operator*=;
%ignore *::operator/;
%ignore *::operator/=;
%ignore *::operator>>;
%ignore *::operator~;
%ignore *::operator==;
%ignore *::operator>;
%ignore *::operator>=;
%ignore *::operator<;
%ignore *::operator<=;
%ignore *::operator|;
%ignore *::operator|=;
%ignore *::operator&;
%ignore *::operator&=;
%ignore *::operator bool;
%ignore *::operator~;
%ignore *::operator();
#endif // SWIGJAVA

#if SWIGOCTAVE 
%ignore *::operator+=;
%ignore *::operator-=;
%ignore *::operator*=;
%ignore *::operator/=;
%ignore *::operator>=;
%ignore *::operator<=;
%ignore *::operator|=;
%ignore *::operator&=;
%rename(or) *::operator|;
#endif // SWIGOCTAVE

#if SWIGRUBY
%ignore *::operator+=;
%ignore *::operator-=;
%ignore *::operator*=;
%ignore *::operator/=;
%ignore *::operator>=;
%ignore *::operator<=;
%ignore *::operator|=;
%ignore *::operator&=;

%ignore *::operator bool;

// Why ?? (ruby error...)
%rename(_allocate) allocate;
#endif // SWIGRUBY



#ifdef SWIGPYTHON
%define PTR_ARG_OUT_APPLY(name)
  %apply unsigned char *OUTPUT{ unsigned char *name };
  %apply unsigned short *OUTPUT{ unsigned short *name };
  %apply unsigned int *OUTPUT{ unsigned int *name };
  %apply char *OUTPUT{ char *name };
  %apply short *OUTPUT{ short *name };
  %apply int *OUTPUT{ int *name };
  %apply char *OUTPUT{const char *name};
%enddef
#else // SWIGPYTHON
%define PTR_ARG_OUT_APPLY(name)
%enddef
#endif // SWIGPYTHON

%define TEMPLATE_WRAP_MACLASS(class, VAR, name)
  %template(name ## _UINT8 ## _ ## VAR) class<UINT8,VAR>;
  %template(name ## _UINT16 ## _ ## VAR) class<UINT16,VAR>;
%enddef


// CMake generated wrap macros

${SWIG_TEMPLATE_WRAP_DEFINITIONS}


// CMake generated includes

${SWIG_COMMON_INCLUDES}
