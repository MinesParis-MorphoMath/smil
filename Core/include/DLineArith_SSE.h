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


#ifndef _D_LINE_ARITH_SSE_H
#define _D_LINE_ARITH_SSE_H



#include <emmintrin.h>


/**
 * \ingroup Core
 * \defgroup Arith
 * @{
 */


template <>
struct addLine<UINT8> : public binaryLineFunctionBase<UINT8>
{
    inline void _exec(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] > (UINT8)(numeric_limits<UINT8>::max()- lIn2[i]) ? numeric_limits<UINT8>::max() : lIn1[i] + lIn2[i];
    }
    inline void _exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
	__m128i r0,r1;
	__m128i *l1 = (__m128i*) lIn1;
	__m128i *l2 = (__m128i*) lIn2;
	__m128i *l3 = (__m128i*) lOut;
	
	unsigned long alignLen = size-size%SIMD_VEC_SIZE;
	
	for(UINT i=0 ; i<alignLen ; i+=16, l1++, l2++, l3++)
	{
	    r0 = _mm_load_si128(l1);
	    r1 = _mm_load_si128(l2);
	    r1 = _mm_adds_epu8(r0, r1);
	    _mm_store_si128(l3, r1);
	}
	
	_exec(lIn1+alignLen, lIn2+alignLen, size%SIMD_VEC_SIZE, lOut+alignLen);
    }
};

template <>
struct addNoSatLine<UINT8> : public binaryLineFunctionBase<UINT8>
{
    inline void _exec(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] + lIn2[i];
    }
    inline void _exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
	__m128i r0,r1;
	__m128i *l1 = (__m128i*) lIn1;
	__m128i *l2 = (__m128i*) lIn2;
	__m128i *l3 = (__m128i*) lOut;
	
	unsigned long alignLen = size-size%SIMD_VEC_SIZE;
	
	for(UINT i=0 ; i<alignLen ; i+=16, l1++, l2++, l3++)
	{
	    r0 = _mm_load_si128(l1);
	    r1 = _mm_load_si128(l2);
	    r1 = _mm_add_epi8(r0, r1);
	    _mm_store_si128(l3, r1);
	}
	
	_exec(lIn1+alignLen, lIn2+alignLen, size%SIMD_VEC_SIZE, lOut+alignLen);
    }
};

template <>
struct subLine<UINT8> : public binaryLineFunctionBase<UINT8>
{
    inline void _exec(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] < (UINT8)(numeric_limits<UINT8>::max() + lIn2[i]) ? numeric_limits<UINT8>::min() : lIn1[i] - lIn2[i];
    }
    inline void _exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
	__m128i r0,r1;
	__m128i *l1 = (__m128i*) lIn1;
	__m128i *l2 = (__m128i*) lIn2;
	__m128i *l3 = (__m128i*) lOut;
	
	unsigned long alignLen = size-size%SIMD_VEC_SIZE;
	
	for(UINT i=0 ; i<alignLen ; i+=16, l1++, l2++, l3++)
	{
	    r0 = _mm_load_si128(l1);
	    r1 = _mm_load_si128(l2);
	    r1 = _mm_subs_epu8(r0, r1);
	    _mm_store_si128(l3, r1);
	}
	
	_exec(lIn1+alignLen, lIn2+alignLen, size%SIMD_VEC_SIZE, lOut+alignLen);
    }
};

template <>
struct subNoSatLine<UINT8> : public binaryLineFunctionBase<UINT8>
{
    inline void _exec(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] - lIn2[i];
    }
    inline void _exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
	__m128i r0,r1;
	__m128i *l1 = (__m128i*) lIn1;
	__m128i *l2 = (__m128i*) lIn2;
	__m128i *l3 = (__m128i*) lOut;
	
	unsigned long alignLen = size-size%SIMD_VEC_SIZE;
	
	for(UINT i=0 ; i<alignLen ; i+=16, l1++, l2++, l3++)
	{
	    r0 = _mm_load_si128(l1);
	    r1 = _mm_load_si128(l2);
	    r1 = _mm_sub_epi8(r0, r1);
	    _mm_store_si128(l3, r1);
	}
	
	_exec(lIn1+alignLen, lIn2+alignLen, size%SIMD_VEC_SIZE, lOut+alignLen);
    }
};

template <>
struct supLine<UINT8> : public binaryLineFunctionBase<UINT8>
{
    inline void _exec(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] > lIn2[i] ? lIn1[i] : lIn2[i];
    }
    inline void _exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
	__m128i r0,r1;
	__m128i *l1 = (__m128i*) lIn1;
	__m128i *l2 = (__m128i*) lIn2;
	__m128i *l3 = (__m128i*) lOut;
	
	unsigned long alignLen = size-size%SIMD_VEC_SIZE;
	
	for(UINT i=0 ; i<alignLen ; i+=16, l1++, l2++, l3++)
	{
	    r0 = _mm_load_si128(l1);
	    r1 = _mm_load_si128(l2);
	    r1 = _mm_max_epu8(r0, r1);
	    _mm_store_si128(l3, r1);
	}
	
	_exec(lIn1+alignLen, lIn2+alignLen, size%SIMD_VEC_SIZE, lOut+alignLen);
    }
};

template <>
struct infLine<UINT8> : public binaryLineFunctionBase<UINT8>
{
    inline void _exec(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] < lIn2[i] ? lIn1[i] : lIn2[i];
    }
    inline void _exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
    {
	__m128i r0,r1;
	__m128i *l1 = (__m128i*) lIn1;
	__m128i *l2 = (__m128i*) lIn2;
	__m128i *l3 = (__m128i*) lOut;
	
	unsigned long alignLen = size-size%SIMD_VEC_SIZE;
	
	for(UINT i=0 ; i<alignLen ; i+=16, l1++, l2++, l3++)
	{
	    r0 = _mm_load_si128(l1);
	    r1 = _mm_load_si128(l2);
	    r1 = _mm_min_epu8(r0, r1);
	    _mm_store_si128(l3, r1);
	}
	
	_exec(lIn1+alignLen, lIn2+alignLen, size%SIMD_VEC_SIZE, lOut+alignLen);
    }
};


/** @}*/

#endif // _D_LINE_ARITH_SSE_H
