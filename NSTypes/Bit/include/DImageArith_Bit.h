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


#ifndef _D_IMAGE_ARITH_BIT_H
#define _D_IMAGE_ARITH_BIT_H


#include "DBitArray.h"
#include "Base/include/private/DImageArith.hpp"

template <>
RES_T copy<Bit>(const Image<Bit> &imIn, Image<Bit> &imOut)
{
    ASSERT_ALLOCATED(&imIn, &imOut);
  
    if (!CHECK_SAME_SIZE(&imIn, &imOut))
	return copy<Bit,Bit>(imIn, 0, 0, 0, imOut, 0, 0, 0);

    typename Image<Bit>::sliceType l1 = imIn.getLines();
    typename Image<Bit>::sliceType l2 = imOut.getLines();

    UINT width = imIn.getWidth();
    
    for (UINT i=0;i<imIn.getLineCount();i++)
      copyLine<Bit>(l1[i], width, l2[i]);

    imOut.modified();
    return RES_OK;
}




#endif // _D_IMAGE_ARITH_BIT_H
