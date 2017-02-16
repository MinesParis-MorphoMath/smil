/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _D_NUMPY_H
#define _D_NUMPY_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"

namespace smil {

template <class T> class Image;

template <class T> int getNumpyType(Image<T> &/*im*/) { return NPY_VOID; } // Default

template <> int getNumpyType(Image<UINT8> &/*im*/) { return NPY_UBYTE; }
template <> int getNumpyType(Image<INT8> &/*im*/) { return NPY_BYTE; }
template <> int getNumpyType(Image<UINT16> &/*im*/) { return NPY_USHORT; }
template <> int getNumpyType(Image<INT16> &/*im*/) { return NPY_SHORT; }
template <> int getNumpyType(Image<UINT32> &/*im*/) { return NPY_UINT; }
template <> int getNumpyType(Image<INT32> &/*im*/) { return NPY_INT; }
template <> int getNumpyType(Image<unsigned long> &/*im*/) { return NPY_ULONG; }
template <> int getNumpyType(Image<long> &/*im*/) { return NPY_LONG; }
template <> int getNumpyType(Image<float> &/*im*/) { return NPY_FLOAT; }
template <> int getNumpyType(Image<double> &/*im*/) { return NPY_DOUBLE; }


// NPY_BOOL=0,
//                     NPY_BYTE, NPY_UBYTE,
//                     NPY_SHORT, NPY_USHORT,
//                     NPY_INT, NPY_UINT,
//                     NPY_LONG, NPY_ULONG,
//                     NPY_LONGLONG, NPY_ULONGLONG,
//                     NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
//                     NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
//                     NPY_OBJECT=17,
//                     NPY_STRING, NPY_UNICODE,
//                     NPY_VOID,
//                     /*

}

#endif // _D_NUMPY_H
