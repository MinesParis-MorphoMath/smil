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


#ifndef _DBIT_H
#define _DBIT_H

#include "DBitArray.h"
#include "DImage_Bit.h"
#include "DLineArith_Bit.h"
#include "DImageArith_Bit.h"
#include "DImageHistogram_Bit.h"
#include "DMorpho_Bit.h"


#include "Base/include/private/DImageMatrix.hpp"

namespace smil
{

    template <>
    void QtImageViewer<Bit>::drawImage();

    template <>
    RES_T VTKImageFileHandler<Bit>::read(const char *filename, Image<Bit> &image)
    {
        return RES_ERR_NOT_IMPLEMENTED;
    }

    template <>
    RES_T VTKImageFileHandler<Bit>::write(const Image<Bit> &image, const char *filename)
    {
        return RES_ERR_NOT_IMPLEMENTED;
    }

    
    template <>
    RES_T matMultiply<Bit>(const Image<Bit> &imIn1, const Image<Bit> &imIn2, Image<Bit> &imOut)
    {
      return RES_ERR_NOT_IMPLEMENTED; 
    }

#if defined SWIGPYTHON and defined USE_NUMPY
    template <>
    PyObject * Image<Bit>::getNumpyArray(bool c_contigous)
    {
    }
#endif // defined SWIGPYTHON and defined USE_NUMPY

} // namespace smil

#endif // _DBIT_H

