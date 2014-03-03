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


#ifndef _D_COLOR_H
#define _D_COLOR_H


#include "DMultichannelTypes.hpp"

#ifdef RGB
#undef RGB
#endif 

namespace smil
{
  
    typedef MultichannelType<UINT8,3> COLOR_UINT8_3;
    typedef MultichannelArray<UINT8,3> COLOR_UINT8_3_Array;
    
    template <>
    struct ImDtTypes< COLOR_UINT8_3 >
    {
	typedef COLOR_UINT8_3 pixelType;
	typedef COLOR_UINT8_3_Array lineType;
	typedef lineType* sliceType;
	typedef sliceType* volType;
	
	typedef MultichannelType<double,3> floatType;

	static inline pixelType min() { return COLOR_UINT8_3(0); }
	static inline pixelType max() { return COLOR_UINT8_3(255); }
	static inline size_t cardinal() { return 256*256*256; }
	static inline lineType createLine(size_t lineLen) 
	{ 
	    lineType arr;
	    arr.createArrays(lineLen);
	    return arr;
	}
	static inline void deleteLine(lineType arr) 
	{ 
	    arr.deleteArrays();
	}
	static inline unsigned long ptrOffset(lineType p, unsigned long n=SIMD_VEC_SIZE) { return (size_t(p.arrays[0])) & (n-1); }
	static inline std::string toString(const COLOR_UINT8_3 &val)
	{
	    stringstream str;
	    str << "(";
	    for (UINT i=0;i<2;i++)
	      str << double(val[i]) << ",";
	    str << double(val[2]) << ")";
	    return str.str();
	}
    };
    
    typedef COLOR_UINT8_3_Array RGBArray;
    
    struct RGB
#ifndef SWIG
    : public COLOR_UINT8_3
#endif // SWIG
    {
	UINT8 &r;
	UINT8 &g;
	UINT8 &b;
	RGB()
	  : r(c[0]), g(c[1]), b(c[2]),
	    MultichannelType<UINT8, 3>(0)
	{
	}
	RGB(const UINT &val)
	  : r(c[0]), g(c[1]), b(c[2]),
	    MultichannelType<UINT8, 3>(val)
	{
	}
	RGB(int _r, int _g, int _b)
	  : r(c[0]), g(c[1]), b(c[2]),
	    MultichannelType<UINT8, 3>(_r,_g,_b)
	{
	}
	RGB(const COLOR_UINT8_3 &rhs)
	  : r(c[0]), g(c[1]), b(c[2]),
	    MultichannelType<UINT8, 3>(rhs)
	{
	}
	RGB& operator =(const RGB &rhs)
	{
	    for (UINT i=0;i<3;i++)
	      c[i] = rhs.value(i);
	    return *this;
	}
    };
    
    
    
    
    template <>
    struct ImDtTypes< RGB > : public ImDtTypes< COLOR_UINT8_3>
    {
	typedef RGB pixelType;
	typedef RGBArray lineType;
	static inline pixelType min() { return RGB(0); }
	static inline pixelType max() { return RGB(255); }
	static inline size_t cardinal() { return 256*256*256; }
    };

    template <> 
    inline const char *getDataTypeAsString(RGB *) { return "RGB"; }

    
    
    typedef MultichannelType<UINT8,4> COLOR_32;
    typedef MultichannelArray<UINT8,4> COLOR_32_Array;
    
    template <>
    struct ImDtTypes< COLOR_32 >
    {
	typedef COLOR_32 pixelType;
	typedef COLOR_32_Array lineType;
	typedef lineType* sliceType;
	typedef sliceType* volType;

	static inline pixelType min() { return COLOR_32(0); }
	static inline pixelType max() { return COLOR_32(255); }
	static inline lineType createLine(size_t lineLen) 
	{ 
	    lineType arr;
	    arr.createArrays(lineLen);
	    return arr;
	}
	static inline void deleteLine(lineType arr) 
	{ 
	    arr.deleteArrays();
	}
	static inline unsigned long ptrOffset(lineType p, unsigned long n=SIMD_VEC_SIZE) { return (size_t(p.arrays[0])) & (n-1); }
	static inline std::string toString(const COLOR_32 &val)
	{
	    stringstream str;
	    for (UINT i=0;i<3;i++)
	      str << double(val[i]) << ", ";
	    str << double(val[3]);
	    return str.str();
	}
    };
    
} // namespace smil

#endif // _D_COLOR_H

