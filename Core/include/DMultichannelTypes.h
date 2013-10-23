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


#ifndef _D_MULTI_CHANNEL_H
#define _D_MULTI_CHANNEL_H


#include "DTypes.h"
#include "DErrors.h"

#include <stdarg.h>

namespace smil
{
    template <class T=UINT8, UINT N=3>
    struct MultichannelArray;
    
    template <class T, UINT N>
    class MultichannelArrayItem;
    
    template <class T, UINT N>
    class MultichannelType
    {
    protected:
	T c[N];
    public:
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
	MultichannelType(const MultichannelArrayItem<T,N> &arrItem)
	{
	}
	
	bool operator ==(const MultichannelType &mc)
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) != mc.value(i))
		return false;
	    return true;
	}
	bool operator !=(const MultichannelType &mc)
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) == mc.value(i))
		return false;
	    return true;
	}
	bool operator !=(const int &val)
	{
	    return this->operator!=(MultichannelType(val));
	}
	bool operator <(const MultichannelType &mc) const
	{
	    for (UINT i=0;i<N;i++)
	      if (this->value(i) > mc.value(i))
		return false;
	    return true;
	}
	bool operator >(const MultichannelType &mc) const
	{
	    return ! (*this)<mc;
	}
	MultichannelType operator -(const MultichannelType &mc)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) - mc.value(i);
	}
	MultichannelType operator -()
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = -this->value(i);
	}
	MultichannelType operator +(const MultichannelType &mc)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) + mc.value(i);
	}
	MultichannelType operator +(const int &val)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      newmc.value(i) = this->value(i) + val;
	}
	MultichannelType operator *(const MultichannelType &mc)
	{
	    MultichannelType newmc;
	    for (UINT i=0;i<N;i++)
	      /*newmc.value(i) = this->value(i) * */mc.value(i);
	}
	inline T& operator[] (UINT i)
	{
	    return this->value(i);
	}
	inline const T& operator[] (UINT i) const
	{
	    return this->value(i);
	}
	
	operator double() const
	{
	    double dval = 0;
	    for (UINT i=0;i<N;i++)
	      dval += c[i];
	    return dval/N;
	}
	operator int() const { return double(*this); }
	operator UINT() const { return double(*this); }
	operator size_t() const { return double(*this); }
	operator UINT8() const { return double(*this); }
	operator UINT16() const { return double(*this); }
//     private:
	virtual const T& value(UINT i) const
	{
	    return c[i];
	}
	virtual T& value(UINT i)
	{
	    return c[i];
	}
    };
    template <class T, UINT N>
    std::ostream& operator<<(std::ostream& stream, const MultichannelType<T,N> &mc)
    {
	for (UINT i=0;i<N-1;i++)
	  stream << double(mc.c[i]) << ", ";
	stream << double(mc.c[N-1]);
	return stream;
    }

    
    template <class T, UINT N>
    class MultichannelArrayItem : public MultichannelType<T,N>
    {
    public:
	typedef MultichannelType<T,N> MCType;
	typedef T* Tptr;
	Tptr _c[N];
	
	MultichannelArrayItem(MultichannelArray<T,N> &mcArray, const UINT &newindex)
	{
	    for (UINT i=0;i<N;i++)
	      _c[i] = &mcArray.arrays[i][newindex];
	}
// 	MultichannelArrayItem(const MCType &mc)
// 	{
// 	    for (UINT i=0;i<N;i++)
// 	      _c[i] = &mc->value(i);
// 	}
// 	MultichannelArrayItem& operator =(const MCType &mc)
// 	{
// 	    for (UINT i=0;i<N;i++)
// 	      *_c[i] = mc[i];
// 	    return *this;
// 	}
//     protected:
	virtual const T& value(UINT i) const
	{
	    return *_c[i];
	}
	virtual T& value(UINT i)
	{
	    return *_c[i];
	}
    };
    
    
    template <class T, UINT N>
    class MultichannelArray
    {
    public:
	friend MultichannelArrayItem<T,N>;
	
	typedef MultichannelType<T,N> MCType;
	
	typedef T *lineType;
	lineType arrays[N];
	
	MultichannelArray()
	  : index(0), size(0), allocatedData(false)
	{
	    resetArrays();
	}
	MultichannelArray(const MultichannelArray &rhs, const UINT &newindex)
	  : index(newindex), size(rhs.size), allocatedData(false)
	{
	    for (UINT i=0;i<N;i++)
	      arrays[i] = rhs.arrays[i];
	}
	MultichannelArray(T *arrayValsPtr, size_t size)
	  : index(0), size(size), allocatedData(false)
	{
	    for (UINT i=0;i<N;i++)
	      arrays[i] = arrayValsPtr + size*i;
	}
	~MultichannelArray()
	{
	    if (allocatedData)
	      deleteArrays();
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
	
	
	
	MultichannelArrayItem<T,N> operator [] (UINT i) // lValue
	{
	    return MultichannelArrayItem<T,N>(*this, i);
	}
	const MCType operator [] (UINT i) const; // rValue
	
	inline MultichannelArrayItem<T,N> operator * ()
	{
	    return MultichannelArrayItem<T,N>(*this, index);
	}
	
	bool operator != (void *ptr)
	{
	    return arrays[0]!=ptr;
	}
	
	MultichannelArray&  operator = (void *ptr)
	{
	    ASSERT((ptr==NULL), "Do not assign pointer to MultichannelArray", *this);
	    
	    deleteArrays();
	    resetArrays();
	    return *this;
	}
	
	inline MultichannelArray operator + (int dp)
	{
	    MultichannelArray ba(*this, this->index + dp);
	    return ba;
	}
	inline MultichannelArray operator + (long unsigned int dp)
	{
	    return operator+((int)dp);
	}
	inline MultichannelArray operator + (UINT dp)
	{
	    return operator+((int)dp);
	}
	MultichannelArray operator - (int dp);
	MultichannelArray& operator ++ (int)
	{
	    index++;
	    return *this;
	}
	size_t operator - (const MultichannelArray<T,N> &arr)
	{
	    return arrays[0]+index - (arr.arrays[0]+arr.index);
	}
// 	MultichannelArray& operator ++ ();
	
// 	MultichannelArray& operator += (int dp) {}
	
	inline MultichannelArray& operator = (const MultichannelArray &rhs)
	{
	    for (UINT i=0;i<N;i++)
	      arrays[i] = rhs.arrays[i];
	    index = rhs.index;
	    size = rhs.size;
	    return *this;
	}
	
	ostream& printSelf(ostream &os=cout);

    protected:
	UINT size;
	UINT index;
	bool allocatedData;
    };
    

   
    
} // namespace smil

#endif // _D_MULTI_CHANNEL_H

