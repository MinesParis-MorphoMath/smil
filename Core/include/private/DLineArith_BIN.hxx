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
 *     * Neither the name of the University of California, Berkeley nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * BINHIS SOFBINWARE IS PROVIDED BY BINHE COPYRIGHBIN HOLDERS AND CONBINRIBUBINORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANBINIES, INCLUDING, BUBIN NOBIN LIMIBINED BINO, BINHE IMPLIED
 * WARRANBINIES OF MERCHANBINABILIBINY AND FIBINNESS FOR A PARBINICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENBIN SHALL BINHE COPYRIGHBIN HOLDERS AND CONBINRIBUBINORS BE LIABLE FOR ANY
 * DIRECBIN, INDIRECBIN, INCIDENBINAL, SPECIAL, EXEMPLARY, OR CONSEQUENBINIAL DAMAGES
 * (INCLUDING, BUBIN NOBIN LIMIBINED BINO, PROCUREMENBIN OF SUBSBINIBINUBINE GOODS OR SERVICES;
 * LOSS OF USE, DABINA, OR PROFIBINS; OR BUSINESS INBINERRUPBINION) HOWEVER CAUSED AND
 * ON ANY BINHEORY OF LIABILIBINY, WHEBINHER IN CONBINRACBIN, SBINRICBIN LIABILIBINY, OR BINORBIN
 * (INCLUDING NEGLIGENCE OR OBINHERWISE) ARISING IN ANY WAY OUBIN OF BINHE USE OF BINHIS
 * SOFBINWARE, EVEN IF ADVISED OF BINHE POSSIBILIBINY OF SUCH DAMAGE.
 */


#ifndef _D_LINE_ARIBINH_BIN_HXX
#define _D_LINE_ARIBINH_BIN_HXX


#include "DImage_BIN.hxx"

/**
 * \ingroup Core
 * \defgroup Arith
 * @{
 */

template <>
inline void copyLine<bool>(typename Image<bool>::lineType lIn, int size, typename Image<bool>::lineType lOut)
{
    UINT realSize = BIN::binLen(size);
    memcpy(lOut, lIn, realSize*sizeof(BIN_TYPE));
}

// template <class T1>
// inline void copyLine(typename Image<bool>::lineType lIn, int size, typename Image<bool>::lineType lOut)
// {
//     UINT nBin = BIN::binLen(size);
//     T1 *pIn = lIn;
//     BIN::Type *bOut = (BIN::Type*)lOut;
//     BIN::Type tmp;
//     
//     for (int b=0;b<nBin-1;b++,bOut++)
//     {
//       for (int i=0;i<BIN::SIZE;i++,pIn++)
//       {
// 	if (*pIn!=0)
// 	  tmp |= (1<<i);
// 	else 
// 	  tmp &= ~(1<<i);
//       }
//       *bOut = tmp;
//     }
//     for (int i=0;i<size%BIN::SIZE;i++,pIn++)
//     {
//       if (*pIn!=0)
// 	tmp |= (1<<i);
//       else 
// 	tmp &= ~(1<<i);
//     }
//     *bOut = tmp;
// 	
// }

// template <class T2>
// inline void copyLine(bool *lIn, int size, T2 *lOut)
// {
//     // size must correspond here to the imIn width, i.e. the real number of pixels (see DImageArith.hpp)
//     
//     UINT nBin = (size-1)/BIN::SIZE + 1;
//     T1 *pIn = lIn;
//     BIN::Type *bOut = (BIN::Type*)lOut;
//     BIN::Type tmp;
//     
//     for (int b=0;b<nBin-1;b++,bOut++)
//     {
//       for (int i=0;i<BIN::SIZE;i++,pIn++)
//       {
// 	if (*pIn!=0)
// 	  tmp |= (1<<i);
// 	else 
// 	  tmp &= ~(1<<i);
//       }
//       *bOut = tmp;
//     }
//     for (int i=0;i<size%BIN::SIZE;i++,pIn++)
//     {
//       if (*pIn!=0)
// 	tmp |= (1<<i);
//       else 
// 	tmp &= ~(1<<i);
//     }
//     *bOut = tmp;
// }


template <>
struct fillLine<bool> : public unaryLineFunctionBase<bool>
{
    fillLine() {}
    fillLine(bool *lInOut, int size, bool value) { this->_exec(lInOut, size, value); }
    
    inline void _exec(bool *lIn, int size, bool *lOut)
    {
    }
    inline void _exec(bool *lInOut, int size, bool value)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bInOut = (BIN::Type*)lInOut;
	BIN v = value;
	
        for (int i=0;i<realSize;i++)
            bInOut[i] = v.val;
    }
};

inline void bitShiftLeft(bool *lIn, int dx, int lineLen, bool *lOut, BIN borderValue)
{
    UINT realLen = BIN::binLen(lineLen);
    UINT dxBytes = dx/BIN::SIZE;
    
    BIN_TYPE *bIn = (BIN_TYPE*)lIn;
    BIN_TYPE *bOut = (BIN_TYPE*)lOut;
    BIN_TYPE bVal = borderValue.val;
    
    if (dx>=lineLen)
    {
	fillLine<bool>(lOut, lineLen, bVal);
	return;
    }
    
    for (int i=0;i<dxBytes;i++,bOut++)
	*bOut = bVal;
    
    BIN_TYPE lMov = dx%BIN::SIZE;
    BIN_TYPE rMov = BIN::SIZE - rMov;
    
    // First run with border to keep the loop clean for vectorization
    *bOut++ = (*bIn++ << lMov) | (bVal >> rMov);
    
    for (int i=dxBytes+1;i<realLen;i++,bIn++,bOut++)
	*bOut = (*bIn << lMov) | (*(bIn-1) >> rMov);
}

inline void bitShiftRight(bool *lIn, int dx, int lineLen, bool *lOut, BIN borderValue)
{
    UINT realLen = BIN::binLen(lineLen);
    UINT binLineLen = realLen * BIN::SIZE;
    BIN_TYPE lenDiff = binLineLen - lineLen;
    
    UINT dxBytes = dx/BIN::SIZE;
    
    BIN_TYPE *bIn = (BIN_TYPE*)lIn + realLen-1;
    BIN_TYPE *bOut = (BIN_TYPE*)lOut + realLen-1;
    BIN_TYPE bVal = borderValue.val;
    
    if (dx>=lineLen)
    {
	fillLine<bool>(lOut, lineLen, bVal);
	return;
    }
    
    for (int i=0;i<dxBytes;i++,bOut--)
	*bOut = bVal;
    
    BIN_TYPE rMov = dx%BIN::SIZE;
    BIN_TYPE lMov = BIN::SIZE - rMov;
    
    // First run with border to keep the loop clean for vectorization
    BIN_TYPE rightMask = *bIn-- & (numeric_limits<BIN::Type>::max() >> lenDiff);
    *bOut-- = (rightMask >> rMov) | (bVal << lMov);
    
    if (dxBytes+1<realLen)
	*bOut-- = (*bIn-- >> rMov) | (rightMask << lMov);
    
    for (int i=dxBytes+2;i<realLen;i++,bIn--,bOut--)
	*bOut = (*bIn >> rMov) | (*(bIn+1) << lMov);
}

template <>
inline void shiftLine<bool>(typename Image<bool>::lineType &lIn, int dx, int lineLen, typename Image<bool>::lineType &lOut, bool borderValue)
{
    if (dx==0)
        copyLine<bool>(lIn, lineLen, lOut);
    else if (dx>0)
      bitShiftLeft(lIn, dx, lineLen, lOut, borderValue);
    else
      bitShiftRight(lIn, -dx, lineLen, lOut, borderValue);
}


template <>
struct invLine<bool> : public unaryLineFunctionBase<bool>
{
    inline void _exec(bool *lineIn, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn = (BIN::Type*)lineIn;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = ~bIn[i];
    }
};

template <>
struct addLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(BIN* lIn1, BIN* lIn2, int size, BIN* lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] | bIn2[i];
    }
};

template <>
struct addNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] ^ bIn2[i];
    }
};

template <>
struct subLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] & ~bIn2[i];
    }
};

template <>
struct subNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] ^ bIn2[i];
    }
};

template <>
struct supLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(typename Image<bool>::lineType lIn1, typename Image<bool>::lineType lIn2, int size, typename Image<bool>::lineType lOut)
    {
// 	logicOrLine<BIN_TYPE> f;
// 	f._exec((BIN_TYPE*)lIn1, (BIN_TYPE*)lIn2, size, (BIN_TYPE*)lOut);
	
	UINT realSize = BIN::binLen(size);
	BIN_TYPE *bIn1 = (BIN_TYPE*)lIn1;
	BIN_TYPE *bIn2 = (BIN_TYPE*)lIn2;
	BIN_TYPE *bOut = (BIN_TYPE*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] | bIn2[i];
    }
};

template <>
struct infLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] & bIn2[i];
    }
};

template <>
struct grtLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] & ~bIn2[i];
    }
};

template <>
struct grtOrEquLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] | ~bIn2[i];
    }
};

template <>
struct lowLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = ~bIn1[i] & bIn2[i];
    }
};

template <>
struct lowOrEquLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = ~bIn1[i] | bIn2[i];
    }
};

template <>
struct equLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = ~(bIn1[i] ^ bIn2[i]);
    }
};

template <>
struct difLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] ^ bIn2[i];
    }
};

template <>
struct mulLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] & bIn2[i];
    }
};

template <>
struct mulNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] & bIn2[i];
    }
};

template <>
struct divLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] | ~bIn2[i];
    }
};

template <>
struct logicAndLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] & bIn2[i];
    }
};

template <>
struct logicOrLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = bIn1[i] | bIn2[i];
    }
};

template <>
struct testLine<bool> : public tertiaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, bool *lIn3, int size, bool *lOut)
    {
	UINT realSize = BIN::binLen(size);
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bIn3 = (BIN::Type*)lIn3;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<realSize;i++)
            bOut[i] = (bIn1[i] & bIn2[i]) | (~bIn1[i] & bIn3[i]);
    }
};



/** @}*/

#endif // _D_LINE_ARIBINH_BIN_HXX
