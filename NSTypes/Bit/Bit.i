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


%include smilCommon.i


%{
#include "DTypes.hpp"
#include "NSTypes/Bit/include/DBit.h"
#include "NSTypes/Bit/include/DBitArray.h"
#include "DImage_Bit.h"
%}


namespace smil
{

    %ignore BitArray::operator[];
    %ignore BitArray::operator++;


    %extend BitArray
    {
	    std::string  __str__() {
		std::stringstream os;
		os << *self;
		return os.str();
	    }

	    bool operator[] (UINT i)
	    {
	    }

    }
}

%include "DBitArray.h"

#ifdef _DIMAGE_HPP
%template(Image_Bit) Image<Bit>;
#endif // _DIMAGE_HPP

#ifdef _D_IMAGE_ARITH_HPP
TEMPLATE_WRAP_FUNC_SINGLE(inv, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(fill, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(randFill, Bit);

TEMPLATE_WRAP_FUNC_2T_FIX_FIRST(copy, Bit);
TEMPLATE_WRAP_FUNC_2T_FIX_SECOND(copy, Bit);

TEMPLATE_WRAP_FUNC_SINGLE(equ, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(diff, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(sup, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(inf, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(low, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(lowOrEqu, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(grt, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(grtOrEqu, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(logicAnd, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(logicOr, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(logicXOr, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(bitAnd, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(bitOr, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(bitXOr, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(test, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(compare, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(mask, Bit);
#endif // _D_IMAGE_ARITH_HPP

#ifdef _D_MEASURES_HPP
TEMPLATE_WRAP_FUNC_SINGLE(vol, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(meanVal, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(area, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(measBarycenter, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(measBoundBox, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(measInertiaMatrix, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(nonZeroOffsets, Bit);
#endif // _D_MEASURES_HPP


#ifdef _D_IMAGE_DRAW_HPP
TEMPLATE_WRAP_FUNC_SINGLE(drawLine, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(drawRectangle, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(drawBox, Bit);
#ifdef USE_FREETYPE
TEMPLATE_WRAP_FUNC_SINGLE(drawText, Bit);
#endif // USE_FREETYPE

TEMPLATE_WRAP_FUNC_SINGLE(vFlip, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(trans, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(resize, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(scale, Bit);
#endif // _D_IMAGE_DRAW_HPP

#ifdef _D_MORPHO_BASE_HPP
TEMPLATE_WRAP_FUNC_SINGLE(dilate, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(erode, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(close, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(open, Bit);
#endif // _D_MORPHO_BASE_HPP

#ifdef _D_MORPHO_GEODESIC_HPP
TEMPLATE_WRAP_FUNC_SINGLE(geoDil, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(geoEro, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(geoBuild, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(geoDualBuild, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(build, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(binBuild, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(dualBuild, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(buildOpen, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(buildClose, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(fillHoles, Bit);
TEMPLATE_WRAP_FUNC_SINGLE(levelPics, Bit);
#endif // _D_MORPHO_GEODESIC_HPP
