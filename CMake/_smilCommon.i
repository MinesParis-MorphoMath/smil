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


%feature("autodoc", "1");

// Language independent exception handler
%include exception.i       

%exception {
    try {
        $action
    } catch(...) {
        SWIG_exception(SWIG_RuntimeError,"Unknown exception");
    }
}


//////////////////////////////////////////////////////////
// Common includes
//////////////////////////////////////////////////////////

// CMake generated includes

${SWIG_COMMON_INCLUDES}

%{
#include "Core/include/DErrors.h"
#include "Core/include/DTypes.h"
#include "Core/include/private/DTypes.hpp"
#include "Core/include/DBaseObject.h"
#include "Core/include/DBaseImage.h"
#include "Core/include/private/DImage.hpp"
#include "Core/include/private/DSharedImage.hpp"
#include "Core/include/private/DInstance.hpp"
#include "Core/include/DCoreInstance.h"
#include "Core/include/DSlot.h"
#include "Core/include/DSignal.h"
#include "Core/include/DCoreEvents.h"
#include "Core/include/private/DGraph.hpp"
using namespace smil;
%}

namespace smil
{
}

using namespace smil;

//////////////////////////////////////////////////////////
// Module definitions
//////////////////////////////////////////////////////////

#ifdef SWIGXML
%define SMIL_MODULE(libname)
    %module(directors="1") libname
%enddef
#endif // SWIGXML

#ifdef SWIGPYTHON
%define SMIL_MODULE(libname)
  %module(package="smilPython", directors="1") libname ## Python
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

#ifdef SWIGPHP
%define SMIL_MODULE(libname)
    %module(directors="1") libname ## Php
%enddef
#endif // SWIGPHP


#define _DCORE
#define _DBASE
#define _DIO
#define _DGUI
#define _DMORPHO

#ifndef SWIGXML
  %include cpointer.i
  %include std_string.i
  %include typemaps.i
#endif // SWIGXML



%rename(__lshift__)  operator<<; 
%ignore *::operator=;


#if defined SWIGJAVA || defined SWIGPHP
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
%ignore *::operator!=;
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
#endif // defined SWIGJAVA || defined SWIGPHP

#ifdef SWIGPHP
%rename(clone) _clone;
%rename(*::clone) _clone;
#endif // SWIGPHP

#if SWIGOCTAVE 
%ignore *::operator!=;
%ignore *::operator+=;
%ignore *::operator-=;
%ignore *::operator*=;
%ignore *::operator/=;
%ignore *::operator>=;
%ignore *::operator<=;
%ignore *::operator|=;
%ignore *::operator&=;
%ignore *::operator|;
%ignore *::operator&;
%ignore *::operator bool;
//%rename(or) *::operator|;
#endif // SWIGOCTAVE

#if SWIGRUBY
%ignore *::operator!=;
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
  %apply size_t *OUTPUT{size_t *name};
  %apply char *OUTPUT{ char *name };
  %apply short *OUTPUT{ short *name };
  %apply int *OUTPUT{ int *name };
  %apply char *OUTPUT{const char *name};
  %apply double *OUTPUT{double *name};
  
  %apply unsigned char *OUTPUT{ unsigned char &name };
  %apply unsigned short *OUTPUT{ unsigned short &name };
  %apply unsigned int *OUTPUT{ unsigned int &name };
  %apply size_t *OUTPUT{size_t &name};
  %apply char *OUTPUT{ char &name };
  %apply short *OUTPUT{ short &name };
  %apply int *OUTPUT{ int &name };
  %apply char *OUTPUT{const char &name};
  %apply double *OUTPUT{double &name};
%enddef
#elif defined SWIGJAVA
%include "arrays_java.i";
%define PTR_ARG_OUT_APPLY(name)
  %typemap(jtype) (unsigned char *name) "byte[]"
%enddef
#else // SWIGPYTHON
%define PTR_ARG_OUT_APPLY(name)
%enddef
#endif // SWIGPYTHON


// CMake generated wrap macros

${SWIG_TEMPLATE_WRAP_DEFINITIONS}


