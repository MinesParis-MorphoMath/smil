/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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
 * BitHIS SOFBitWARE IS PROVIDED BY BitHE COPYRIGHBit HOLDERS AND CONBitRIBUBitORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANBitIES, INCLUDING, BUBit NOBit LIMIBitED BitO, BitHE IMPLIED
 * WARRANBitIES OF MERCHANBitABILIBitY AND FIBitNESS FOR A PARBitICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENBit SHALL BitHE COPYRIGHBit HOLDERS AND CONBitRIBUBitORS BE LIABLE FOR ANY
 * DIRECBit, INDIRECBit, INCIDENBitAL, SPECIAL, EXEMPLARY, OR CONSEQUENBitIAL DAMAGES
 * (INCLUDING, BUBit NOBit LIMIBitED BitO, PROCUREMENBit OF SUBSBitIBitUBitE GOODS OR SERVICES;
 * LOSS OF USE, DABitA, OR PROFIBitS; OR BUSINESS INBitERRUPBitION) HOWEVER CAUSED AND
 * ON ANY BitHEORY OF LIABILIBitY, WHEBitHER IN CONBitRACBit, SBitRICBit LIABILIBitY, OR BitORBit
 * (INCLUDING NEGLIGENCE OR OBitHERWISE) ARISING IN ANY WAY OUBit OF BitHE USE OF BitHIS
 * SOFBitWARE, EVEN IF ADVISED OF BitHE POSSIBILIBitY OF SUCH DAMAGE.
 */


#ifndef _D_LINE_ARITH_RGB_H
#define _D_LINE_ARITH_RGB_H


#include "Base/include/private/DLineArith.hpp"
#include "Core/include/DColor.h"

/**
 * \ingroup Arith
 * @{
 */


namespace smil
{
    template <>
    inline void copyLine<RGB>(const Image<RGB>::lineType lIn, const size_t size, Image<RGB>::lineType lOut)
    {
	for (UINT n=0;n<3;n++)
	  memcpy(lOut.arrays[n], lIn.arrays[n], size*sizeof(UINT8));
    }

    template <class T1>
    RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY, size_t startZ, size_t sizeX, size_t sizeY, size_t sizeZ, Image<RGB> &imOut, size_t outStartX=0, size_t outStartY=0, size_t outStartZ=0)
    {
	return RES_ERR;
    }
    
    template <>
    inline void shiftLine(const Image<RGB>::lineType lIn, int dx, size_t lineLen, Image<RGB>::lineType lOut, RGB borderValue)
    {
	for (UINT n=0;n<3;n++)
	    shiftLine<UINT8>(lIn.arrays[n], dx, lineLen, lOut.arrays[n], borderValue[n]);
    }
    
    template <>
    struct fillLine<RGB> : public unaryLineFunctionBase<RGB>
    {
	typedef Image<RGB>::lineType lineType;
	fillLine() {}
	fillLine(const lineType lIn, const size_t size, const RGB value) { this->_exec(lIn, size, value); }
	
	virtual void _exec(const lineType lIn, const size_t size, lineType lOut)
	{
	    copyLine<RGB>(lIn, size, lOut);
	}
	virtual void _exec(lineType lInOut, const size_t size, const RGB value)
	{
	    for (UINT n=0;n<3;n++)
	    {
		UINT8 *cArr = lInOut.arrays[n];
		UINT8 val = value[n];
		for (size_t i=0;i<size;i++)
		    cArr[i] = val;
	    }
	    
	}
    };

    
    template <>
    double vol(const Image<RGB> &imIn);
    
//     template <>
//     std::map<RGB, UINT> histogram(const Image<RGB> &imIn);

    
    template <>
    struct supLine<RGB> : public binaryLineFunctionBase<RGB>
    {
	typedef Image<RGB>::lineType lineType;
	inline void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
	{
	    for (UINT n=0;n<3;n++)
	    {
		UINT8 *cArrIn1 = lIn1.arrays[n];
		UINT8 *cArrIn2 = lIn2.arrays[n];
		UINT8 *cArrOut = lOut.arrays[n];
		
		for (size_t i=0;i<size;i++)
		    cArrOut[i] = cArrIn1[i] > cArrIn2[i] ? cArrIn1[i] : cArrIn2[i];
	    }
	}
    };

    template <>
    struct infLine<RGB> : public binaryLineFunctionBase<RGB>
    {
	typedef Image<RGB>::lineType lineType;
	virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
	{
	    for (UINT n=0;n<3;n++)
	    {
		UINT8 *cArrIn1 = lIn1.arrays[n];
		UINT8 *cArrIn2 = lIn2.arrays[n];
		UINT8 *cArrOut = lOut.arrays[n];
		
		for (size_t i=0;i<size;i++)
		    cArrOut[i] = cArrIn1[i] < cArrIn2[i] ? cArrIn1[i] : cArrIn2[i];
	    }
	}
    };

    template <>
    struct equLine<RGB> : public binaryLineFunctionBase<RGB>
    {
	equLine() 
	  : trueVal(numeric_limits<UINT8>::max()), falseVal(0) {}
	  
	UINT8 trueVal, falseVal;
	  
	typedef Image<RGB>::lineType lineType;
	virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
	{
	    for (UINT n=0;n<3;n++)
	    {
		UINT8 *cArrIn1 = lIn1.arrays[n];
		UINT8 *cArrIn2 = lIn2.arrays[n];
		UINT8 *cArrOut = lOut.arrays[n];
		
		for (size_t i=0;i<size;i++)
		    cArrOut[i] = cArrIn1[i] == cArrIn2[i] ? trueVal : falseVal;
	    }
	}
    };
    
} // namespace smil

/** @}*/

#endif // _D_LINE_ARITH_RGB_H
