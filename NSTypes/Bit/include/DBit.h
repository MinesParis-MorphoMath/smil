/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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

#include "Morpho/include/private/DMorphoArrow.hpp"
#include "Morpho/include/private/DMorphImageOperations.hpp"

template <>
void _DGUI QtImageViewer<Bit>::drawImage();

template <>
RES_T readVTK<Bit>(const char *filename, Image<Bit> &image)
{
}

template <>
RES_T writeVTK<Bit>(const Image<Bit> &image, const char *filename, bool binary)
{
}

template <class lineFunction_T>
class unaryMorphArrowImageFunction<Bit, lineFunction_T>
{
public:
    typedef Image<Bit> imageType;
    unaryMorphArrowImageFunction(Bit b=0) {}
    inline RES_T operator()(const imageType &imIn, imageType &imOut, const StrElt &se) {  }
    RES_T _exec_single(const Image<Bit> &imIn, Image<Bit> &imOut, const StrElt &se) {}
    
};


#endif // _DBIT_H

