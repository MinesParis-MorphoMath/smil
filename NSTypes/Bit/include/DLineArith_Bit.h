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


#ifndef _D_LINE_ARITH_BIT_H
#define _D_LINE_ARITH_BIT_H


#include "DBitArray.h"
#include "Base/include/private/DLineArith.hpp"

/**
 * \ingroup Arith
 * @{
 */

namespace smil
{
    
    template <>
    inline void copyLine<Bit>(const typename Image<Bit>::lineType lIn, const size_t size, typename Image<Bit>::lineType lOut)
    {
    //     copyLine<BitArray::INT_TYPE>(lIn.intArray, BitArray::INT_SIZE(size), lOut.intArray);
    //     UINT realSize = BitArray::INT_SIZE(size)*sizeof(BitArray::INT_TYPE);
    //       memcpy(&lOut, &lIn, realSize);

        if (lIn.index==0 && lOut.index==0)
        {
            UINT fullNbr = size/BitArray::INT_TYPE_SIZE; 
            UINT bitRes  = size - fullNbr*BitArray::INT_TYPE_SIZE;
            
            memcpy(lOut.intArray, lIn.intArray, fullNbr*sizeof(BitArray::INT_TYPE));
            
            if (bitRes)
              lOut.intArray[fullNbr] = lIn.intArray[fullNbr];
    //           lOut.intArray[fullNbr] = ((lIn.intArray[fullNbr] >> (BitArray::INT_TYPE_SIZE-bitRes))) | ((lOut.intArray[fullNbr] << bitRes));
        }
        else
        {
            for (size_t i=0;i<size;i++)
              lOut[i] = lIn[i];
        }
    }

    template <>
    struct fillLine<Bit> : public unaryLineFunctionBase<Bit>
    {
        fillLine() {}
        fillLine(BitArray lInOut, size_t size, Bit value) { this->_exec(lInOut, size, value); }
        
        inline void _exec(BitArray lIn, size_t size, BitArray lOut) { copyLine<Bit>(lIn, size, lOut); }
        inline void _exec(BitArray lInOut, size_t size, Bit value)
        {
            BitArray::INT_TYPE intVal = (bool)value ? BitArray::INT_TYPE_MAX() : BitArray::INT_TYPE_MIN();
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
              lInOut.intArray[i] = intVal;
        }
    };
    // template <class T1>
    // inline void copyLine(T1 *lIn, size_t size, BitArray lOut)
    // {
    //     UINT nBin = Bit::binLen(size);
    //     T1 *pIn = lIn;
    //     Bit::Type *bOut = (Bit::Type*)lOut;
    //     Bit::Type tmp;
    //     
    //     for (int b=0;b<nBin-1;b++,bOut++)
    //     {
    //       for (int i=0;i<Bit::SIZE;i++,pIn++)
    //       {
    //         if (*pIn!=0)
    //           tmp |= (1<<i);
    //         else 
    //           tmp &= ~(1<<i);
    //       }
    //       *bOut = tmp;
    //     }
    //     for (int i=0;i<size%Bit::SIZE;i++,pIn++)
    //     {
    //       if (*pIn!=0)
    //         tmp |= (1<<i);
    //       else 
    //         tmp &= ~(1<<i);
    //     }
    //     *bOut = tmp;
    //         
    // }

    // template <class T2>
    // inline void copyLine(BitArray lIn, int size, T2 *lOut)
    // {
    //     // size must correspond here to the imIn width, i.e. the real number of pixels (see DImageArith.hpp)
    //     
    //     UINT nBin = (size-1)/Bit::SIZE + 1;
    //     T1 *pIn = lIn;
    //     Bit::Type *bOut = (Bit::Type*)lOut;
    //     Bit::Type tmp;
    //     
    //     for (int b=0;b<nBin-1;b++,bOut++)
    //     {
    //       for (int i=0;i<Bit::SIZE;i++,pIn++)
    //       {
    //         if (*pIn!=0)
    //           tmp |= (1<<i);
    //         else 
    //           tmp &= ~(1<<i);
    //       }
    //       *bOut = tmp;
    //     }
    //     for (int i=0;i<size%Bit::SIZE;i++,pIn++)
    //     {
    //       if (*pIn!=0)
    //         tmp |= (1<<i);
    //       else 
    //         tmp &= ~(1<<i);
    //     }
    //     *bOut = tmp;
    // }



    inline void bitShiftLeft(BitArray lIn, int dx, size_t lineLen, BitArray lOut, Bit borderValue)
    {
        UINT realLen = BitArray::INT_SIZE(lineLen);
        UINT dxBytes = dx/BitArray::INT_TYPE_SIZE;
        
        BitArray::INT_TYPE *bIn = lIn.intArray;
        BitArray::INT_TYPE *bOut = lOut.intArray;
        BitArray::INT_TYPE bBorder = (bool)borderValue ? BitArray::INT_TYPE_MAX() : BitArray::INT_TYPE_MIN();
        
        if (dx>=(int)lineLen)
        {
            fillLine<Bit>(lOut, lineLen, borderValue);
            return;
        }
        
        for (int i=0;i<(int)dxBytes;i++,bOut++)
            *bOut = bBorder;
        
        UINT lMov = dx%BitArray::INT_TYPE_SIZE;
        UINT rMov = BitArray::INT_TYPE_SIZE - lMov;
        
        // First run with border to keep the loop clean for vectorization
        *bOut++ = (*bIn++ << lMov) | (bBorder >> rMov);
        
        for (int i=dxBytes+1;i<(int)realLen;i++,bIn++,bOut++)
            *bOut = (*bIn << lMov) | (*(bIn-1) >> rMov);
    }

    inline void bitShiftRight(BitArray lIn, int dx, size_t lineLen, BitArray lOut, Bit borderValue)
    {
        UINT realLen = BitArray::INT_SIZE(lineLen);
        UINT lenDiff = lIn.getBitPadX();
        UINT dxReal = dx + lenDiff;
        
        UINT dxBytes = dxReal/BitArray::INT_TYPE_SIZE;
        
        BitArray::INT_TYPE *bIn = lIn.intArray + realLen-1;
        BitArray::INT_TYPE *bOut = lOut.intArray + realLen-1;
        BitArray::INT_TYPE bBorder = (bool)borderValue ? BitArray::INT_TYPE_MAX() : BitArray::INT_TYPE_MIN();
        
        if (dx>=(int)lineLen)
        {
            fillLine<Bit>(lOut, lineLen, borderValue);
            return;
        }
        
        for (int i=0;i<(int)dxBytes;i++,bOut--)
            *bOut = bBorder;
        
        BitArray::INT_TYPE rMov = dx%BitArray::INT_TYPE_SIZE;
        BitArray::INT_TYPE lMov = BitArray::INT_TYPE_SIZE - rMov;
        
        // First run with border to keep the loop clean for vectorization
        BitArray::INT_TYPE rightMask = *bIn-- & (BitArray::INT_TYPE_MAX() >> lenDiff);
        *bOut-- = (rightMask >> rMov) | (bBorder << lMov);
        
        if (dxBytes+1<realLen)
            *bOut-- = (*bIn-- >> rMov) | (rightMask << lMov);
        
        for (int i=dxBytes+2;i<(int)realLen;i++,bIn--,bOut--)
            *bOut = (*bIn >> rMov) | (*(bIn+1) << lMov);
    }

    template <>
    inline void shiftLine<Bit>(const Image<Bit>::lineType lIn, int dx, size_t lineLen, Image<Bit>::lineType lOut, Bit borderValue)
    {
        if (dx==0)
            copyLine<Bit>(lIn, lineLen, lOut);
        else if (dx>0)
          bitShiftLeft(lIn, dx, lineLen, lOut, borderValue);
        else
          bitShiftRight(lIn, -dx, lineLen, lOut, borderValue);
    }


    template <>
    struct invLine<Bit> : public unaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = ~(lIn.intArray[i]);
        }
    };

    template <>
    struct addLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] | lIn2.intArray[i];
        }
    };

    template <>
    struct addNoSatLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] ^ lIn2.intArray[i];
        }
    };

    template <>
    struct subLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] & ~lIn2.intArray[i];
        }
    };

    template <>
    struct subNoSatLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] ^ lIn2.intArray[i];
        }
    };

    template <>
    struct supLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        typedef typename Image<Bit>::lineType lineType;
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = (lIn1.intArray[i] | lIn2.intArray[i]);
        }
    };

    template <>
    struct infLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] & lIn2.intArray[i];
        }
    };

    template <>
    struct grtLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] & ~lIn2.intArray[i];
        }
    };

    template <>
    struct grtOrEquLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] | ~lIn2.intArray[i];
        }
    };

    template <>
    struct lowLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = ~lIn1.intArray[i] & lIn2.intArray[i];
        }
    };

    template <>
    struct lowOrEquLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = ~lIn1.intArray[i] | lIn2.intArray[i];
        }
    };

    template <>
    struct equLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        typedef typename Image<Bit>::lineType lineType;
        inline void _exec(lineType lIn1, lineType lIn2, size_t size, lineType lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = ~(lIn1.intArray[i] ^ lIn2.intArray[i]);
        }
    };

    template <>
    struct diffLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] ^ lIn2.intArray[i];
        }
    };

    template <>
    struct mulLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] & lIn2.intArray[i];
        }
    };

    template <>
    struct mulNoSatLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] & lIn2.intArray[i];
        }
    };

    template <>
    struct divLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] | ~lIn2.intArray[i];
        }
    };

    template <>
    struct logicAndLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] & lIn2.intArray[i];
        }
    };

    template <>
    struct logicOrLine<Bit> : public binaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = lIn1.intArray[i] | lIn2.intArray[i];
        }
    };

    template <>
    struct testLine<Bit, Bit> : public tertiaryLineFunctionBase<Bit>
    {
        inline void _exec(BitArray lIn1, BitArray lIn2, BitArray lIn3, size_t size, BitArray lOut)
        {
            for (size_t i=0;i<BitArray::INT_SIZE(size);i++)
                lOut.intArray[i] = (lIn1.intArray[i] & lIn2.intArray[i]) | (~lIn1.intArray[i] & lIn3.intArray[i]);
        }
    };

} // namespace smil

/** @}*/

#endif // _D_LINE_ARITH_BIT_H
