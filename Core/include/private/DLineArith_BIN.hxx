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


inline void bitShiftLeft(BIN *lIn, int dx, int lineLen, BIN *lOut, BIN borderValue)
{
    UINBIN dxBytes = dx/BIN::SIZE;
    UINBIN bitLen = lineLen * BIN::SIZE;

    
    if (dx>=bitLen)
    {
	for (int i=0;i<lineLen;i++)
	  lOut[i].val = borderValue.val;
	return;
    }
    
    if (dxBytes > 0)
    {
	BIN *l1 = lIn;
	BIN *l2 = lOut + dxBytes;
	
        for(int i=dxBytes;i<lineLen;i++, l1++, l2++)
            l2->val = l1->val;
	
        for(int i=0;i<dxBytes;i++)
	    lOut[i].val = borderValue.val;
    }
    else
      memcpy(lOut, lIn, lineLen*sizeof(BIN));

    for(int i=0;i<dx%BIN::SIZE;i++)
    {
        for (int j=lineLen-1;j>=0;j--)
        {
            lOut[j].val <<= 1;
	    
            if (lOut[j].val & BIN::MS_BIBIN)
	      lOut[j+1].val |= 0x01;
	}
    }

}

inline void bitShiftRight(BIN *lIn, int dx, int lineLen, BIN *lOut, BIN borderValue)
{
    UINBIN dxBytes = dx/BIN::SIZE;
    UINBIN bitLen = lineLen * BIN::SIZE;

    if (dx>=bitLen)
    {
	for (int i=0;i<lineLen;i++)
	  lOut[i].val = borderValue.val;
	return;
    }
    
    if (dxBytes > 0)
    {
	BIN *l1 = lIn + dxBytes;
	BIN *l2 = lOut;
	
        for(int i=0;i<lineLen-dxBytes;i++,l1++,l2++)
            l2->val = l1->val;
	
        for(int i=lineLen-dxBytes;i<lineLen;i++)
	    lOut[i].val = borderValue.val;
    }
    else
      memcpy(lOut, lIn, lineLen*sizeof(BIN));

    for(int i=0;i<dx%BIN::SIZE;i++)
    {
        for (int j=0;j<lineLen;j++)
        {
            lOut[j].val >>= 1;
	    
            if (lOut[j+1].val & 0x01)
	      lOut[j].val |= BIN::MS_BIBIN;
	}
    }

}

template <>
inline void shiftLine<BIN>(BIN *lIn, int dx, int lineLen, BIN *lOut, BIN borderValue)
{
    if (dx==0)
        copyLine(lIn, lineLen, lOut);
    else if (dx>0)
      bitShiftLeft(lIn, dx, lineLen, lOut, borderValue);
    else
      bitShiftRight(lIn, dx, lineLen, lOut, borderValue);
}


// template <>
// struct invLine : public unaryLineFunctionBase<BIN>
// {
//     inline void _exec(BIN *lineIn, int size, BIN *lOut)
//     {
//         for (int i=0;i<size;i++)
//             lOut[i] = ~lineIn[i];
//     }
// };

template <>
struct addLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN* lIn1, BIN* lIn2, int size, BIN* lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] | lIn2[i];
    }
};

template <>
struct addNoSatLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] ^ lIn2[i];
    }
};

template <>
struct subLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] & ~lIn2[i];
    }
};

template <>
struct subNoSatLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] ^ lIn2[i];
    }
};

template <>
struct supLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] > lIn2[i] ? lIn1[i] : lIn2[i];
    }
    inline void _exec_aligned(BIN *lIn1, BIN *lIn2, int size, BIN *lOut) {
        _exec(lIn1, lIn2, size, lOut);
    }
};

template <>
struct infLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] < lIn2[i] ? lIn1[i] : lIn2[i];
    }
    inline void _exec_aligned(BIN *lIn1, BIN *lIn2, int size, BIN *lOut) {
        _exec(lIn1, lIn2, size, lOut);
    }
};

template <>
struct grtLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] > lIn2[i] ? numeric_limits<BIN>::max() : 0;
    }
};

template <>
struct grtOrEquLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] >= lIn2[i] ? numeric_limits<BIN>::max() : 0;
    }
};

template <>
struct lowLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] < lIn2[i] ? numeric_limits<BIN>::max() : 0;
    }
};

template <>
struct lowOrEquLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] <= lIn2[i] ? numeric_limits<BIN>::max() : 0;
    }
};

template <>
struct equLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] == lIn2[i] ? numeric_limits<BIN>::max() : 0;
    }
};

template <>
struct difLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] != lIn2[i] ? numeric_limits<BIN>::max() : 0;
    }
};


template <>
struct mulLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = double(lIn1[i]) * double(lIn2[i]) > double(numeric_limits<BIN>::max()) ? numeric_limits<BIN>::max() : lIn1[i] * lIn2[i];
    }
    inline void _exec_aligned(BIN *lIn1, BIN *lIn2, int size, BIN *lOut) {
        _exec(lIn1, lIn2, size, lOut);
    }
};

template <>
struct mulNoSatLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = (BIN)(lIn1[i] * lIn2[i]);
    }
};

template <>
struct divLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
        {
            lOut[i] = lIn2[i]==0 ? numeric_limits<BIN>::max() : lIn1[i] / lIn2[i];
        }
    }
};

template <>
struct logicAndLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = (BIN)(lIn1[i] && lIn2[i]);
    }
};

template <>
struct logicOrLine : public binaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = (BIN)(lIn1[i] || lIn2[i]);
    }
};


template <>
struct testLine : public tertiaryLineFunctionBase<BIN>
{
    inline void _exec(BIN *lIn1, BIN *lIn2, BIN *lIn3, int size, BIN *lOut)
    {
        for (int i=0;i<size;i++)
        {
            lOut[i] = lIn1[i] ? lIn2[i] : lIn3[i];
        }
    }
};



/** @}*/

#endif // _D_LINE_ARIBINH_BIN_HXX
