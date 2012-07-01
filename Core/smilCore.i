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
%module(directors="1") smilCorePython
#endif // SWIGPYTHON

#ifdef SWIGJAVA
%module(directors="0") smilCoreJava 
// Problemes de directors avec Java... (a resoudre)
#endif // SWIGJAVA

#ifdef SWIGOCTAVE
%module(directors="1") smilCoreOctave
#endif // SWIGOCTAVE

%include smilCommon.i



%{
/* Includes the header in the wrapper code */
#include "DCommon.h"
#include "DSignal.h"
#include "DSlot.h"
#include "DImage.h"
#include "DTypes.hpp"
#include "DBaseObject.h"
#include "DCoreInstance.h"
#include "DBaseImageOperations.hpp"
#include "DBaseLineOperations.hpp"
#include "DImageArith.h"
#include "DImageDraw.hpp"
#include "DImageTransform.hpp"
#include "DLineHistogram.hpp"
#include "DImageHistogram.hpp"
/*#include "D_BaseOperations.h"*/
#include "memory"
#include <listobject.h>
#include "DCoreEvents.h"

%}

%include "carrays.i"
%array_class(double, dArray);
//%array_class(void, voidArray);
%array_class(UINT8, uint8Array);
 

%{
UINT8 *createArray(int size)
{
  return (UINT8*)malloc(size_t(size));
}
void printArray(UINT8 *arr, int size)
{
  for (int i=0;i<size;i++)
{
  cout << (int)arr[i] << endl;
}
}
%}

UINT8 *createArray(int size);
void printArray(UINT8 *arr, int size);



///// Numpy /////
#if defined SWIGPYTHON && defined USE_NUMPY
%{
    #include "DNumpy.h"
%}

%extend Image 
{
	PyObject * getNumArray()
	{
	    npy_intp d[] = { self->getHeight(), self->getWidth(), self->getDepth() }; // axis are inverted...
	    PyObject *array = PyArray_SimpleNewFromData(self->getDimension(), d, getNumpyType(*self), self->getPixels());
	    
	    npy_intp t[] = { 1, 0, 2 };
	    PyArray_Dims trans_dims;
	    trans_dims.ptr = t;
	    trans_dims.len = self->getDimension();
	    
	    PyObject *res = PyArray_Transpose((PyArrayObject*) array, &trans_dims);
	    Py_DECREF(array);
	    return res;
	}
}

%init
%{
	import_array();
%}
#endif // defined SWIGPYTHON && defined USE_NUMPY


%extend Image 
{
	std::string  __str__() 
	{
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
}

%rename(Index) operator[](UINT i);

#ifdef SMIL_WRAP_Bit
%extend BitArray
{
	std::string  __str__() {
	    std::stringstream os;
	    os << *self;
	    return os.str();
	}
/*
	bool operator[] (UINT i)
	{
	}
*/
}
%ignore BitArray::operator++;
#endif // SMIL_WRAP_Bit

PTR_ARG_OUT_APPLY(ret_min)
PTR_ARG_OUT_APPLY(ret_max)
PTR_ARG_OUT_APPLY(w)
PTR_ARG_OUT_APPLY(h)
PTR_ARG_OUT_APPLY(d)

%include std_vector.i

%include "DCommon.h"
%include "DTypes.hpp"
%include "DBaseObject.h"
%include "DCoreInstance.h"
%include "DBaseImage.h"
%include "DImage.hpp"
%include "DImage.hxx"
%include "DImage.h"

#ifdef SMIL_WRAP_Bit
%include "DBitArray.h"
#endif // SMIL_WRAP_Bit

%include "DTypes.hpp"
/*%include "D_BaseOperations.h" */
%include "DBaseImageOperations.hpp"
%include "DBaseLineOperations.hpp"
%include "DLineArith.h"
%include "DLineArith.hpp"
%include "DImageArith.hpp"
%include "DImageDraw.hpp"
%include "DLineHistogram.hpp"
%include "DImageHistogram.hpp"

%include "DImageViewer.hpp"

#ifndef SWIGJAVA
// generate directors for Signal and Slot (for virtual methods overriding)
%feature("director") Signal;
%feature("director") Slot;
%feature("director") baseImageSlot;
#endif // SWIGJAVA

%include "DSignal.h"
%include "DSlot.h"
%include "DCoreEvents.h"

%template(baseImageSlot) Slot<baseImageEvent>;

%template(objVector) vector<baseObject*>;
%template(uintVector) vector<UINT>;

TEMPLATE_WRAP_CLASS(imageViewer);
TEMPLATE_WRAP_CLASS(Image);

TEMPLATE_WRAP_FUNC(createImage);

TEMPLATE_WRAP_FUNC_CROSS2(copy);

TEMPLATE_WRAP_FUNC(inv);
TEMPLATE_WRAP_FUNC(fill);
TEMPLATE_WRAP_FUNC(add);
TEMPLATE_WRAP_FUNC(addNoSat);
TEMPLATE_WRAP_FUNC(sub);
TEMPLATE_WRAP_FUNC(subNoSat);
TEMPLATE_WRAP_FUNC(mul);
TEMPLATE_WRAP_FUNC(mulNoSat);
TEMPLATE_WRAP_FUNC(div);

TEMPLATE_WRAP_FUNC(equ);
TEMPLATE_WRAP_FUNC(diff);
TEMPLATE_WRAP_FUNC(sup);
TEMPLATE_WRAP_FUNC(inf);
TEMPLATE_WRAP_FUNC(low);
TEMPLATE_WRAP_FUNC(lowOrEqu);
TEMPLATE_WRAP_FUNC(grt);
TEMPLATE_WRAP_FUNC(grtOrEqu);
TEMPLATE_WRAP_FUNC(logicAnd);
TEMPLATE_WRAP_FUNC(logicOr);
TEMPLATE_WRAP_FUNC(test);

TEMPLATE_WRAP_FUNC(vol);
TEMPLATE_WRAP_FUNC(minVal);
TEMPLATE_WRAP_FUNC(maxVal);
TEMPLATE_WRAP_FUNC(rangeVal);


TEMPLATE_WRAP_FUNC(histo);

TEMPLATE_WRAP_FUNC(thresh);
TEMPLATE_WRAP_FUNC(stretchHist);
TEMPLATE_WRAP_FUNC(enhanceContrast);

TEMPLATE_WRAP_FUNC(drawRectangle);

%include "DImageTransform.hpp"

TEMPLATE_WRAP_FUNC(vFlip);
TEMPLATE_WRAP_FUNC(trans);
TEMPLATE_WRAP_FUNC(resize);



