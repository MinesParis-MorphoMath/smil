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


#ifndef _DBINARY_BIN_H
#define _DBINARY_BIN_H

#include <cstring>
#include <iostream>

#include <limits>

#include "DTypes.hpp"
#include "DMemory.hpp"

using namespace std;

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

#ifdef USE_64BIT_IDS
//     typedef UINT8 INT_TYPE;
    typedef UINT64 BIN_TYPE;
#else    
    typedef UINT32 BIN_TYPE;
#endif // USE_64BIT_IDS 

struct bitIndex
{
    BIN_TYPE *byte;
    unsigned int index;
    
    bitIndex& operator=(bool val)
    {
      if (val)
	(*byte) |= (1UL<<index);
      else
	(*byte) &= ~(1UL<<index);
      return *this;
    }
    operator bool()
    {
      return (*byte) & (1UL<<index);
    }
};

struct BIN
{
    BIN_TYPE val;
    
    BIN(BIN_TYPE v = numeric_limits<BIN_TYPE>::min()) : val(v) {}
    BIN(bool b) : val(b ? this->max() : this->min()) {}
    BIN(double v) : val(v==0 ? this->min() : this->max()) {}
    
    static const BIN_TYPE SIZE = sizeof(BIN_TYPE)*CHAR_BIT;
    
    static inline BIN_TYPE min() { return numeric_limits<BIN_TYPE>::min(); }
    static inline BIN_TYPE max() { return numeric_limits<BIN_TYPE>::max(); }
    
    //! Most significant bit
    static const BIN_TYPE MS_BIT = (1UL << (SIZE - 2));
    //! Less significant bit
    static const BIN_TYPE LS_BIT = 0x01;
    
    
    typedef BIN_TYPE Type;
    typedef Type *lineType;
    typedef lineType *sliceType;
    
    static inline BIN_TYPE binLen(BIN_TYPE bitCount) { return (bitCount-1)/BIN::SIZE + 1; }

    inline bitIndex& operator[] (UINT8 pos)
    {
	static bitIndex b;
	b.byte = &val;
	b.index = pos;
 	return b;
    }
    ostream& printSelf(ostream &os=cout)
    {
	for (int i=0;i<SIZE;i++)
	  os << this->operator[](i) << " ";
	return os;
    }
    inline BIN& operator=(BIN_TYPE v)
    {
	val = v;
	return *this;
    }
    inline BIN& operator=(bool b)
    {
	val = b ? this->max() : this->min();
	return *this;
    }
    inline BIN& operator=(const char* s)
    {
	UINT iMax = strlen(s) < SIZE ? strlen(s) : SIZE;
	
	val = 0;
	for (int i=0;i<iMax;i++)
	  val += (s[i]-48) * (1<<i);
	return *this;
    }
//     inline operator bool()
//     {
// 	return val!=0;
//     }
};

inline ostream& operator << (ostream &os, BIN &b)
{
    return b.printSelf(os);
}

// #define BIN BIN<UINT32>



// 
// inline RES_T setPixel(UINT x, UINT y, UINT z, pixelType value)
// {
//     if (x>=width || y>=height || z>=depth)
// 	return RES_ERR;
//     pixels[z*width*height+y*width+x] = value;
//     modified();
//     return RES_OK;
// }
// inline RES_T setPixel(UINT x, UINT y, pixelType value)
// {
//     if (x>=width || y>=height)
// 	return RES_ERR;
//     pixels[height+y*width+x] = value;
//     modified();
//     return RES_OK;
// }
// inline RES_T setPixel(UINT offset, pixelType value)
// {
//     if (offset >= pixelCount)
// 	return RES_ERR;
//     pixels[offset] = value;
//     modified();
//     return RES_OK;
// }



#endif // _DBINARY_BIN_H
