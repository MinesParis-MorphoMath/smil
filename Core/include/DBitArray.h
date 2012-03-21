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


#ifndef _DBITARRAY_H
#define _DBITARRAY_H

#include "DBinary.hpp"
#include "DTypes.hpp"
// #include <qshareddata.h>

class Bit;

class BitArray
{
public:

#ifdef USE_64BIT_IDS
//     typedef UINT8 INT_TYPE;
    typedef UINT64 INT_TYPE;
#else    
    typedef UINT32 INT_TYPE;
#endif // USE_64BIT_IDS 
    

    INT_TYPE *intArray;
    
    
    static const INT_TYPE INT_TYPE_SIZE = sizeof(INT_TYPE)*CHAR_BIT;
    static inline INT_TYPE INT_TYPE_MIN() { return numeric_limits<INT_TYPE>::min(); }
    static inline INT_TYPE INT_TYPE_MAX() { return numeric_limits<INT_TYPE>::max(); }
    static inline UINT INT_SIZE(UINT bitCount) { return (bitCount-1)/INT_TYPE_SIZE + 1; }
    
    //! Most significant bit
    static const INT_TYPE INT_MS_BIT = (1UL << (INT_TYPE_SIZE - 2));
    //! Less significant bit
    static const INT_TYPE INT_LS_BIT = 0x01;
    
    
    BitArray()
            : index(0), intArray(NULL), bitWidth(0), intWidth(0), height(0)
    {}
    BitArray(const BitArray &rhs)
    {
	this->setSize(rhs.bitWidth, rhs.height);
	this->intArray = rhs.intArray;
	this->index = rhs.index;
    }
    BitArray(UINT _bitWidth, UINT _bitHeight=1)
            : index(0), intArray(NULL)
    {
        setSize(_bitWidth, _bitHeight);
    }
    BitArray(INT_TYPE *arr, UINT _bitWidth, UINT _bitHeight=1)
            : index(0), intArray(arr)
    {
        setSize(_bitWidth, _bitHeight);
    }
    ~BitArray()
    {
	intArray = NULL;
    }
    
    
    inline UINT getBitWidth() { return bitWidth; }
    inline UINT getIntWidth() { return intWidth; }
    inline UINT getIntNbr() { return intWidth*height; }
    inline UINT getHeight() { return height; }
    inline UINT getBitPadX() { return intWidth*INT_TYPE_SIZE - bitWidth; }

    UINT index;
    
    void setSize(UINT _bitWidth, UINT _bitHeight=1);
    void createIntArray()
    {
	if (!intArray)
	  intArray = createAlignedBuffer<INT_TYPE>(intWidth*height);
    }
    void deleteIntArray()
    {
	if (intArray)
	  deleteAlignedBuffer<INT_TYPE>(intArray);
	intArray = NULL;
    }
    
    bool getValue(UINT ind);
    void setValue(UINT ind, bool val);
    Bit operator [] (UINT i);
    Bit operator * ();
    BitArray operator + (int dp);
    BitArray operator - (int dp);
    BitArray& operator ++ (int);
    BitArray& operator ++ ();
    
    BitArray& operator = (const BitArray &rhs)
    {
	this->setSize(rhs.bitWidth, rhs.height);
	this->intArray = rhs.intArray;
	this->index = rhs.index;
	return *this;
    }
    
    ostream& printSelf(ostream &os=cout);

private:
    UINT intWidth;
    UINT bitWidth;
    UINT height;
};


inline ostream& operator << (ostream &os, BitArray &b)
{
    return b.printSelf(os);
}

class Bit
{
public:
    Bit() : bitArray(NULL), value(false), index(0) {}
    Bit(bool v) : bitArray(NULL), value(v), index(0) {}
    BitArray *bitArray;
    UINT index;
    bool value;
    operator bool();
    Bit& operator = (bool v);
    Bit& operator = (Bit &src);
};


inline void BitArray::setSize(UINT _bitWidth, UINT _bitHeight)
{
    bitWidth = _bitWidth;
    intWidth = INT_SIZE(bitWidth);
    height = _bitHeight;
}

inline bool BitArray::getValue(UINT ind)
{
    int Y = ind / bitWidth;
    int X = (ind + Y*this->getBitPadX()) / INT_TYPE_SIZE;
    int x = (ind-Y*bitWidth) % INT_TYPE_SIZE;
    return (intArray[X] & (1UL << x))!=0;
}

inline void BitArray::setValue(UINT ind, bool val)
{
    int Y = ind / bitWidth;
    int X = (ind + Y*this->getBitPadX()) / INT_TYPE_SIZE;
    int x = (ind-Y*bitWidth) % INT_TYPE_SIZE;
    if (val)
        intArray[X] |= (1UL << x);
    else intArray[X] &= ~(1UL << x);
}

inline Bit BitArray::operator [] (UINT i)
{
    Bit b;
    b.bitArray = this;
    b.index = i;
    return b;
}

inline Bit BitArray::operator * ()
{
    Bit b;
    b.bitArray = this;
    b.index = index;
    return b;
}

inline BitArray BitArray::operator+(int dp)
{
    BitArray ba(this->intArray, this->bitWidth, this->height);
    ba.index = this->index + dp;
    return ba;
}

inline BitArray BitArray::operator-(int dp)
{
    BitArray ba(this->intArray, this->bitWidth, this->height);
    ba.index = this->index - dp;
    return ba;
}

inline BitArray& BitArray::operator++(int)
{
    index++;
    return *this;
}

inline BitArray& BitArray::operator++()
{
    index++;
    return *this;
}


inline ostream& BitArray::printSelf(ostream &os)
{
    if (!this->intArray)
      return os;
    
    for (int i=0;i<bitWidth;i++)
    {
      os << this->operator[](i);
      if (i<bitWidth-1)
      {
	  if ((i+1)%INT_TYPE_SIZE==0) 
	    os << "-";
	  else
	    os << " ";
      }
    }
    return os;
}



inline Bit::operator bool()
{
    if (bitArray)
	return bitArray->getValue(index);
    else return value;
}

inline Bit& Bit::operator = (bool v)
{
    if (bitArray)
        bitArray->setValue(index, v);
    else value = v;
    return *this;
}

inline Bit& Bit::operator = (Bit &src)
{
    if (bitArray)
    {
        if (src.bitArray)
            bitArray->setValue(index, src.bitArray->getValue(index));
        else
            bitArray->setValue(index, src.value);
    }
    else
    {
        if (src.bitArray)
            value = src.bitArray->getValue(index);
        else
            value = src.value;
    }
    return *this;
}

template <>
inline const char *getDataTypeAsString(Bit &val)
{
    return "Bit";
}

#endif // _DBITARRAY_H

