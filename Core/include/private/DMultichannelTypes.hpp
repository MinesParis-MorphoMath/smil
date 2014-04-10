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


#ifndef _D_MULTI_CHANNEL_TYPES_H
#define _D_MULTI_CHANNEL_TYPES_H


#include "DTypes.h"
#include "DErrors.h"

#include <stdarg.h>

namespace smil
{
    template <class T=UINT8, UINT N=3>
    class MultichannelArray;
    
    template <class T, UINT N>
    class MultichannelArrayItem;
    
//     template <class T, UINT N>
//     class VirtualMultichannelType
    template <class T, UINT N>
    class MultichannelType
    {
    protected:
	T c[N];
    public:
	typedef T DataType;
	static UINT channelNumber() { return N; }
	
	MultichannelType()
	{
	    for (UINT i=0;i<N;i++)
	      c[i] = 0;
	}
	MultichannelType(const T &val)
	{
	    for (UINT i=0;i<N;i++)
	      c[i] = val;
	}
	MultichannelType(const T &_r, const T &_g, const T &_b)
	{
	    c[0] = _r;
	    c[1] = _g;
	    c[2] = _b;
	}
	MultichannelType(const MultichannelType<T,N> &mc)
	{
	    for (UINT i=0;i<N;i++)
	      c[i] = mc.value(i);
	}

	MultichannelType& operator =(const MultichannelType &mc)
	{
	    for (UINT i=0;i<N;i++)
	      this->value(i) = mc.value(i);
	    return *this;
	}
	
	bool operator ==(const MultichannelType &mc) const
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) != mc.value(i))
		return false;
	    return true;
	}
	bool operator ==(const int &val) const
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) != val)
		return false;
	    return true;
	}
	bool operator !=(const MultichannelType &mc) const
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) == mc.value(i))
		return false;
	    return true;
	}
	bool operator !=(const int &val) const
	{
	    return this->operator!=(MultichannelType(val));
	}
	bool operator <(const MultichannelType &mc) const
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) >= mc.value(i))
		return false;
	    return true;
	}
	bool operator <=(const MultichannelType &mc) const
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) > mc.value(i))
		return false;
	    return true;
	}
	bool operator >(const MultichannelType &mc) const
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) <= mc.value(i))
		return false;
	    return true;
	}
	bool operator >=(const MultichannelType &mc) const
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) < mc.value(i))
		return false;
	    return true;
	}
	MultichannelType operator ~()
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = ~this->value(i);
	    return newmc;
	}
#ifndef SWIG
	MultichannelType operator !()
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = !this->value(i);
	    return newmc;
	}
#endif // SWIG
	MultichannelType operator -(const MultichannelType &mc) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) - mc.value(i);
	    return newmc;
	}
	MultichannelType operator -(const int &val) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) - val;
	    return newmc;
	}
	MultichannelType operator -(const size_t &val) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) - val;
	    return newmc;
	}
	MultichannelType operator -()
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = -this->value(i);
	    return newmc;
	}
	MultichannelType operator +(const MultichannelType &mc) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) + mc.value(i);
	    return newmc;
	}
	MultichannelType operator +(const int &val) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) + val;
	    return newmc;
	}
	MultichannelType& operator +=(const MultichannelType &mc)
	{
	    for (UINT i=0;i<N;i++)
	      this->value(i) += mc.value(i);
	    return *this;
	}
	MultichannelType& operator -=(const MultichannelType &mc)
	{
	    for (UINT i=0;i<N;i++)
	      this->value(i) -= mc.value(i);
	    return *this;
	}
	MultichannelType operator *(const MultichannelType &mc) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) * mc.value(i);
	    return newmc;
	}
	MultichannelType operator *(const double &val) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) * val;
	    return newmc;
	}
	MultichannelType operator *(const size_t &val) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) * val;
	    return newmc;
	}
	MultichannelType operator /(const MultichannelType &mc) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) / mc.value(i);
	    return newmc;
	}
	MultichannelType operator /(const double &val) const
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) / val;
	    return newmc;
	}
	MultichannelType operator /(const size_t &val)  const { return this->operator/(double(val)); }
	MultichannelType operator &(const MultichannelType &mc)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) & mc.value(i);
	    return newmc;
	}
	MultichannelType operator ^(const MultichannelType &mc)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) ^ mc.value(i);
	    return newmc;
	}
	MultichannelType operator |(const MultichannelType &mc)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) | mc.value(i);
	    return newmc;
	}
	MultichannelType operator |=(const MultichannelType &mc)
	{
	    for (UINT i=0;i<N;i++)
	      this->value(i) |= mc.value(i);
	    return *this;
	}
#ifndef SWIG
	MultichannelType& operator ++(int) // postfix
	{
	    for (UINT i=0;i<N;i++)
	      this->value(i) ++;
	    return *this;
	}
	MultichannelType operator &&(const MultichannelType &mc)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) && mc.value(i);
	    return newmc;
	}
	MultichannelType operator ||(const MultichannelType &mc)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) || mc.value(i);
	    return newmc;
	}
	inline T& operator[] (UINT i)
	{
	    return this->value(i);
	}
	inline const T& operator[] (UINT i) const
	{
	    return this->value(i);
	}
#endif // SWIG

	operator double() const
	{
	    double dval = 0;
	    for (UINT i=0;i<N;i++)
	      dval += c[i];
	    return dval/N;
	}
	operator int() const { return double(*this); }
	operator UINT() const { return double(*this); }
#ifdef USE_64BIT_IDS
	operator size_t() const { return double(*this); }
#endif // USE_64BIT_IDS
	operator UINT8() const { return double(*this); }
	operator UINT16() const { return double(*this); }
	operator bool() const { return double(*this); }
	operator signed char() const { return double(*this); }
	operator char() const { return double(*this); }
	operator long int() const { return static_cast<long int>(double(*this)); }

	virtual const T& value(const UINT &i) const
	{
	    return c[i];
	}
	virtual T& value(const UINT &i)
	{
	    return c[i];
	}
    };
    
    template <class T, UINT N>
    std::ostream& operator<<(std::ostream& stream, const MultichannelType<T,N> &mc)
    {
	for (UINT i=0;i<N-1;i++)
	  stream << double(mc.value(i)) << ", ";
	stream << double(mc.value(N-1));
	return stream;
    }

    
    template <class T, UINT N>
    class MultichannelArrayItem : public MultichannelType<T,N>
    {
    public:
	typedef MultichannelType<T,N> MCType;
	typedef T* Tptr;
	Tptr _c[N];
	
	MultichannelArrayItem(const MultichannelArray<T,N> &mcArray, const UINT &index)
	{
	    for (UINT i=0;i<N;i++)
	      _c[i] = &mcArray.arrays[i][index];
	}
	MultichannelArrayItem& operator =(const MCType &mc)
	{
	    for (UINT i=0;i<N;i++)
	      *_c[i] = mc.value(i);
	    return *this;
	}
	virtual const T& value(const UINT &i) const
	{
	    return *_c[i];
	}
	virtual T& value(const UINT &i)
	{
	    return *_c[i];
	}
    };
    
    
    template <class T, UINT N>
    class MultichannelArray
    {
    public:
// 	friend MultichannelArrayItem<T,N>;
	
	typedef MultichannelType<T,N> MCType;
	
	typedef T *lineType;
	lineType arrays[N];
	
	MultichannelArray()
	  : size(0), index(0), allocatedData(false)
	{
	    resetArrays();
	}
	MultichannelArray(const MultichannelArray &rhs)
	  : size(rhs.size-rhs.index), index(0), allocatedData(false)
	{
	    for (UINT i=0;i<N;i++)
	      arrays[i] = rhs.arrays[i] + rhs.index;
	}
	MultichannelArray(const MultichannelArray &rhs, const UINT &newindex)
	  : size(rhs.size-rhs.index-newindex), index(0), allocatedData(false)
	{
	    for (UINT i=0;i<N;i++)
	      arrays[i] = rhs.arrays[i] + rhs.index + newindex;
	}
	MultichannelArray(T *arrayValsPtr, size_t size)
	  : size(size), index(0), allocatedData(false)
	{
	    for (UINT i=0;i<N;i++)
	      arrays[i] = arrayValsPtr + size*i;
	}
	~MultichannelArray()
	{
// 	    if (allocatedData)
// 	      deleteArrays();
	}
	
	
	void resetArrays()
	{
	    for (UINT i=0;i<N;i++)
	      arrays[i] = NULL;
	}
	
	bool isAllocated() { return allocatedData; }
	
	void createArrays(UINT len)
	{
	    if (allocatedData)
	      deleteArrays();
	    for (UINT i=0;i<N;i++)
	      arrays[i] = createAlignedBuffer<T>(len);
	    size = len;
	    allocatedData = true;
	}
	void deleteArrays()
	{
	    if (!allocatedData)
	      return;
	    for (UINT i=0;i<N;i++)
	      deleteAlignedBuffer<T>(arrays[i]);
	    size = 0;
	    allocatedData = false;
	}
	
	
	
#ifndef SWIG
	MultichannelArrayItem<T,N> operator [] (UINT i) // lValue
	{
	    return MultichannelArrayItem<T,N>(*this, index+i);
	}
	const MCType operator [] (UINT i) const // rValue
	{
	    return MultichannelArrayItem<T,N>(*this, index+i);
	}
#endif // SWIG
	
	inline MultichannelArrayItem<T,N> operator * ()
	{
	    return MultichannelArrayItem<T,N>(*this, index);
	}
	
	bool operator != (void *ptr)
	{
	    return arrays[0]!=ptr;
	}
	bool operator == (void *ptr)
	{
	    return arrays[0]==ptr;
	}
	
	MultichannelArray&  operator = (void *ptr)
	{
	    ASSERT((ptr==NULL), "Do not assign pointer to MultichannelArray", *this);
	    
	    deleteArrays();
	    resetArrays();
	    return *this;
	}
	
	inline MultichannelArray operator + (int dp) const
	{
	    MultichannelArray ba(*this, this->index + dp);
	    return ba;
	}
	inline MultichannelArray operator + (size_t dp) const
	{
	    return operator+((int)dp);
	}
	MultichannelArray operator - (int dp);
#ifndef SWIG
	MultichannelArray& operator ++ (int)
	{
	    index++;
	    return *this;
	}
#endif // SWIG
	size_t operator - (const MultichannelArray<T,N> &arr)
	{
	    return arrays[0]+index - (arr.arrays[0]+arr.index);
	}
// 	MultichannelArray& operator ++ ();
	
// 	MultichannelArray& operator += (int dp) {}
	
	inline MultichannelArray& operator = (const MultichannelArray &rhs)
	{
	    for (UINT i=0;i<N;i++)
	      arrays[i] = rhs.arrays[i] + rhs.index;
	    index = 0;
	    size = rhs.size - rhs.index;
	    allocatedData = rhs.allocatedData;
	    return *this;
	}
	operator void *()
	{
		return arrays;
	}
	operator char *()
	{
		return arrays;
	}
	
	ostream& printSelf(ostream &os=cout);

    protected:
	UINT size;
	UINT index;
	bool allocatedData;
    };
    

    
} // namespace smil

#endif // _D_MULTI_CHANNEL_TYPES_H

