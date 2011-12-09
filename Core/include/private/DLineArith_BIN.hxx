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

// template <>
// inline void copyLine<bool,bool>(bool *lIn, int size, bool *lOut)
// {
//     memcpy(lOut, lIn, size*sizeof(BIN_TYPE));
// }

template <class T1>
inline void copyLine(T1 *lIn, int size, bool *lOut)
{
    // size must correspond here to the imIn width, i.e. the real number of pixels (see DImageArith.hpp)
    
    UINT nBin = (size-1)/BIN::SIZE + 1;
    T1 *pIn = lIn;
    BIN::Type *bOut = (BIN::Type*)lOut;
    BIN::Type tmp;
    
    for (int b=0;b<nBin-1;b++,bOut++)
    {
      for (int i=0;i<BIN::SIZE;i++,pIn++)
      {
	if (*pIn!=0)
	  tmp |= (1<<i);
	else 
	  tmp &= ~(1<<i);
      }
      *bOut = tmp;
    }
    for (int i=0;i<size%BIN::SIZE;i++,pIn++)
    {
      if (*pIn!=0)
	tmp |= (1<<i);
      else 
	tmp &= ~(1<<i);
    }
    *bOut = tmp;
	
}

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
    inline void _exec(bool *lInOut, int size, bool value)
    {
	BIN::Type *bInOut = (BIN::Type*)lInOut;
	BIN v = value;
	
        for (int i=0;i<size;i++)
            bInOut[i] = value;
    }
};

inline void bitShiftLeft(bool *lIn, int dx, int lineLen, bool *lOut, BIN borderValue)
{
    UINT dxBytes = dx/BIN::SIZE;
    UINT bitLen = lineLen * BIN::SIZE;
    
    BIN::Type *bIn = (BIN::Type*)lIn;
    BIN::Type *bOut = (BIN::Type*)lOut;
    BIN::Type bVal = borderValue.val;
    
    if (dx>=bitLen)
    {
	for (int i=0;i<lineLen;i++)
	  bOut[i] = bVal;
	return;
    }
    
    if (dxBytes > 0)
    {
	BIN::Type *l1 = bIn;
	BIN::Type *l2 = bOut + dxBytes;
	
        for(int i=dxBytes;i<lineLen;i++, l1++, l2++)
            *l2 = *l1;
	
        for(int i=0;i<dxBytes;i++)
	    bOut[i] = bVal;
    }
    else
      memcpy(bOut, bIn, lineLen*sizeof(BIN));

    for(int i=0;i<dx%BIN::SIZE;i++)
    {
        for (int j=lineLen-1;j>=0;j--)
        {
            bOut[j] <<= 1;
	    
            if (bOut[j] & BIN::MS_BIT)
	      bOut[j+1] |= 0x01;
	}
    }

}

inline void bitShiftRight(bool *lIn, int dx, int lineLen, bool *lOut, BIN borderValue)
{
    UINT dxBytes = dx/BIN::SIZE;
    UINT bitLen = lineLen * BIN::SIZE;
    
    BIN::Type *bIn = (BIN::Type*)lIn;
    BIN::Type *bOut = (BIN::Type*)lOut;
    BIN::Type bVal = borderValue.val;

    if (dx>=bitLen)
    {
	for (int i=0;i<lineLen;i++)
	  bOut[i] = bVal;
	return;
    }
    
    if (dxBytes > 0)
    {
	BIN::Type *l1 = bIn + dxBytes;
	BIN::Type *l2 = bOut;
	
        for(int i=0;i<lineLen-dxBytes;i++,l1++,l2++)
            *l2 = *l1;
	
        for(int i=lineLen-dxBytes;i<lineLen;i++)
	    bOut[i] = bVal;
    }
    else
      memcpy(bOut, bIn, lineLen*sizeof(BIN_TYPE));

    for(int i=0;i<dx%BIN::SIZE;i++)
    {
        for (int j=0;j<lineLen;j++)
        {
            bOut[j] >>= 1;
	    
            if (bOut[j+1] & 0x01)
	      bOut[j] |= BIN::MS_BIT;
	}
    }

}

template <>
inline void shiftLine<bool>(bool *lIn, int dx, int lineLen, bool *lOut, bool borderValue)
{
    if (dx==0)
        copyLine<bool,bool>(lIn, lineLen, lOut);
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
	BIN::Type *bIn = (BIN::Type*)lineIn;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = ~bIn[i];
    }
};

template <>
struct addLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(BIN* lIn1, BIN* lIn2, int size, BIN* lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] | bIn2[i];
    }
};

template <>
struct addNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] ^ bIn2[i];
    }
};

template <>
struct subLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] & ~bIn2[i];
    }
};

template <>
struct subNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] ^ bIn2[i];
    }
};

template <>
struct supLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
// 	logicOrLine<BIN_TYPE> f;
// 	f._exec((BIN_TYPE*)lIn1, (BIN_TYPE*)lIn2, size, (BIN_TYPE*)lOut);
	
	BIN_TYPE *bIn1 = (BIN_TYPE*)lIn1;
	BIN_TYPE *bIn2 = (BIN_TYPE*)lIn2;
	BIN_TYPE *bOut = (BIN_TYPE*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] | bIn2[i];
    }
};

template <>
struct infLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] & bIn2[i];
    }
};

template <>
struct grtLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] & ~bIn2[i];
    }
};

template <>
struct grtOrEquLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] | ~bIn2[i];
    }
};

template <>
struct lowLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = ~bIn1[i] & bIn2[i];
    }
};

template <>
struct lowOrEquLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = ~bIn1[i] | bIn2[i];
    }
};

template <>
struct equLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = ~(bIn1[i] ^ bIn2[i]);
    }
};

template <>
struct difLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] ^ bIn2[i];
    }
};

template <>
struct mulLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] & bIn2[i];
    }
};

template <>
struct mulNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] & bIn2[i];
    }
};

template <>
struct divLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] | ~bIn2[i];
    }
};

template <>
struct logicAndLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] & bIn2[i];
    }
};

template <>
struct logicOrLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = bIn1[i] | bIn2[i];
    }
};

template <>
struct testLine<bool> : public tertiaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, bool *lIn3, int size, bool *lOut)
    {
	BIN::Type *bIn1 = (BIN::Type*)lIn1;
	BIN::Type *bIn2 = (BIN::Type*)lIn2;
	BIN::Type *bIn3 = (BIN::Type*)lIn3;
	BIN::Type *bOut = (BIN::Type*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i] = (bIn1[i] & bIn2[i]) | (~bIn1[i] & bIn3[i]);
    }
};



/** @}*/

#endif // _D_LINE_ARIBINH_BIN_HXX
