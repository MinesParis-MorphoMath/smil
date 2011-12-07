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


inline void bitShiftLeft(bool *lIn, int dx, int lineLen, bool *lOut, BIN borderValue)
{
    UINT dxBytes = dx/BIN::SIZE;
    UINT bitLen = lineLen * BIN::SIZE;
    
    BIN *bIn = (BIN*)lIn;
    BIN *bOut = (BIN*)lOut;
    
    if (dx>=bitLen)
    {
	for (int i=0;i<lineLen;i++)
	  bOut[i].val = borderValue.val;
	return;
    }
    
    if (dxBytes > 0)
    {
	BIN *l1 = bIn;
	BIN *l2 = bOut + dxBytes;
	
        for(int i=dxBytes;i<lineLen;i++, l1++, l2++)
            l2->val = l1->val;
	
        for(int i=0;i<dxBytes;i++)
	    bOut[i].val = borderValue.val;
    }
    else
      memcpy(bOut, bIn, lineLen*sizeof(BIN));

    for(int i=0;i<dx%BIN::SIZE;i++)
    {
        for (int j=lineLen-1;j>=0;j--)
        {
            bOut[j].val <<= 1;
	    
            if (bOut[j].val & BIN::MS_BIT)
	      bOut[j+1].val |= 0x01;
	}
    }

}

inline void bitShiftRight(bool *lIn, int dx, int lineLen, bool *lOut, BIN borderValue)
{
    UINT dxBytes = dx/BIN::SIZE;
    UINT bitLen = lineLen * BIN::SIZE;
    
    BIN *bIn = (BIN*)lIn;
    BIN *bOut = (BIN*)lOut;

    if (dx>=bitLen)
    {
	for (int i=0;i<lineLen;i++)
	  bOut[i].val = borderValue.val;
	return;
    }
    
    if (dxBytes > 0)
    {
	BIN *l1 = bIn + dxBytes;
	BIN *l2 = bOut;
	
        for(int i=0;i<lineLen-dxBytes;i++,l1++,l2++)
            l2->val = l1->val;
	
        for(int i=lineLen-dxBytes;i<lineLen;i++)
	    bOut[i].val = borderValue.val;
    }
    else
      memcpy(bOut, bIn, lineLen*sizeof(BIN_TYPE));

    for(int i=0;i<dx%BIN::SIZE;i++)
    {
        for (int j=0;j<lineLen;j++)
        {
            bOut[j].val >>= 1;
	    
            if (bOut[j+1].val & 0x01)
	      bOut[j].val |= BIN::MS_BIT;
	}
    }

}

template <>
inline void shiftLine<bool>(bool *lIn, int dx, int lineLen, bool *lOut, bool borderValue)
{
    if (dx==0)
        copyLine(lIn, lineLen, lOut);
    else if (dx>0)
      bitShiftLeft(lIn, dx, lineLen, lOut, borderValue);
    else
      bitShiftRight(lIn, dx, lineLen, lOut, borderValue);
}


template <>
struct invLine<bool> : public unaryLineFunctionBase<bool>
{
    inline void _exec(bool *lineIn, int size, bool *lOut)
    {
	BIN *bIn = (BIN*)lineIn;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = ~bIn[i].val;
    }
};

template <>
struct addLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(BIN* lIn1, BIN* lIn2, int size, BIN* lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val | bIn2[i].val;
    }
};

template <>
struct addNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val ^ bIn2[i].val;
    }
};

template <>
struct subLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val & ~bIn2[i].val;
    }
};

template <>
struct subNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val ^ bIn2[i].val;
    }
};

template <>
struct supLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val | bIn2[i].val;
    }
};

template <>
struct infLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val & bIn2[i].val;
    }
};

template <>
struct grtLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val & ~bIn2[i].val;
    }
};

template <>
struct grtOrEquLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val | ~bIn2[i].val;
    }
};

template <>
struct lowLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = ~bIn1[i].val & bIn2[i].val;
    }
};

template <>
struct lowOrEquLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = ~bIn1[i].val | bIn2[i].val;
    }
};

template <>
struct equLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = ~(bIn1[i].val ^ bIn2[i].val);
    }
};

template <>
struct difLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val ^ bIn2[i].val;
    }
};

template <>
struct mulLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val & bIn2[i].val;
    }
};

template <>
struct mulNoSatLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val & bIn2[i].val;
    }
};

template <>
struct divLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val | ~bIn2[i].val;
    }
};

template <>
struct logicAndLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val & bIn2[i].val;
    }
};

template <>
struct logicOrLine<bool> : public binaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = bIn1[i].val | bIn2[i].val;
    }
};

template <>
struct testLine<bool> : public tertiaryLineFunctionBase<bool>
{
    inline void _exec(bool *lIn1, bool *lIn2, bool *lIn3, int size, bool *lOut)
    {
	BIN *bIn1 = (BIN*)lIn1;
	BIN *bIn2 = (BIN*)lIn2;
	BIN *bIn3 = (BIN*)lIn3;
	BIN *bOut = (BIN*)lOut;
	
        for (int i=0;i<size;i++)
            bOut[i].val = (bIn1[i].val & bIn2[i].val) | (~bIn1[i].val & bIn3[i].val);
    }
};



/** @}*/

#endif // _D_LINE_ARIBINH_BIN_HXX
