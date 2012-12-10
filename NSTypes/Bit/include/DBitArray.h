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


#ifndef _DBITARRAY_H
#define _DBITARRAY_H

#include <iostream>

#include "DTypes.hpp"

class BitArray;

class Bit
{
public:
    Bit() : bitArray(NULL), value(false), index(0) {}
    Bit(bool v) : bitArray(NULL), value(v), index(0) {}
    BitArray *bitArray;
    UINT index;
    bool value;
    operator bool() const;
    Bit& operator = (const bool v);
    Bit& operator = (const Bit &src);
    inline bool operator< (const Bit &src) const { return value<src.value; }
};


class BitArray
{
public:


    typedef size_t INT_TYPE;

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
    BitArray(bool *arr, UINT _bitWidth, UINT _bitHeight=1)
            : index(0), intArray(NULL)
    {
        setSize(_bitWidth, _bitHeight);
	createIntArray();
	for (size_t i=0;i<_bitWidth;i++)
	  setValue(i, arr[i]);
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
    operator bool() { return intArray!=NULL; }
    Bit operator [] (UINT i); // lValue
    Bit operator [] (UINT i) const; // rValue
    
    inline Bit operator * ()
    {
      Bit b;
      b.bitArray = this;
      b.index = index;
      return b;
    }
    
//     operator void* () { return (void*)this->intArray; }
//     operator char* () { return (char*)this->intArray; }
    void operator=(void *ptr) { this->intArray = (INT_TYPE*)ptr; }
    inline BitArray operator + (int dp)
    {
	BitArray ba(this->intArray, this->bitWidth, this->height);
	ba.index = this->index + dp;
	return ba;
    }
    inline BitArray operator + (long unsigned int dp)
    {
	return operator+((int)dp);
    }
    inline BitArray operator + (UINT dp)
    {
	return operator+((int)dp);
    }
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


template <>
inline const char *getDataTypeAsString(Bit &val)
{
    return "Bit";
}


template <>
struct ImDtTypes<Bit>
{
    typedef Bit pixelType;
    typedef BitArray lineType;
    typedef lineType* sliceType;
    typedef sliceType* volType;

    static inline pixelType min() { return Bit(0); }
    static inline pixelType max() { return Bit(1); }
    static inline lineType createLine(UINT lineLen) 
    { 
	BitArray ba(lineLen);
	ba.createIntArray();
	return ba; 
    }
    static inline void deleteLine(lineType line) 
    { 
	line.deleteIntArray();
    }
    static inline unsigned long ptrOffset(lineType p, unsigned long n=SIMD_VEC_SIZE) { return ((unsigned long)(p.intArray)) & (n-1); }
};


#endif // _DBITARRAY_H

