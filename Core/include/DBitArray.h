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

class Bit;

class BitArray
{
public:
    BitArray()
            : index(0), intArray(NULL), bitWidth(0), intWidth(0), height(0)
    {}
    BitArray(UINT _bitWidth, UINT _bitHeight=1)
            : index(0), intArray(NULL)
    {
        setSize(_bitWidth, _bitHeight);
    }
    BitArray(BIN_TYPE *arr, UINT _bitWidth, UINT _bitHeight=1)
            : index(0), intArray(arr)
    {
        setSize(_bitWidth, _bitHeight);
    }
    
    BIN_TYPE *intArray;
    
    UINT getBitWidth() { return bitWidth; }
    UINT getIntWidth() { return intWidth; }
    UINT getIntNbr() { return intWidth*height; }
    UINT getHeight() { return height; }

    void setSize(UINT _bitWidth, UINT _bitHeight=1);
    bool getValue(UINT ind);
    void setValue(UINT ind, bool val);
    Bit operator [] (UINT i);
    Bit operator * ();
    BitArray& operator ++ (int);
    BitArray& operator ++ ();
private:
    UINT intWidth;
    UINT bitWidth;
    UINT height;
    
    UINT bitPadX;
    UINT index;
};

class Bit
{
public:
    Bit() : bitArray(NULL), value(false) {}
    Bit(bool v) : bitArray(NULL), value(v) {}
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
    intWidth = BIN::binLen(bitWidth);
    bitPadX = intWidth*BIN::SIZE - bitWidth;
    height = _bitHeight;
}

inline bool BitArray::getValue(UINT ind)
{
    int Y = ind / bitWidth;
    int X = (ind + Y*bitPadX) / BIN::SIZE;
    int x = ind % BIN::SIZE;
    return (intArray[X] & (1UL << x))!=0;
}

inline void BitArray::setValue(UINT ind, bool val)
{
    int Y = ind / bitWidth;
    int X = (ind + Y*bitPadX) / BIN::SIZE;
    int x = ind % BIN::SIZE;
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


inline Bit::operator bool()
{
    return bitArray->getValue(index);
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


#endif // _DBITARRAY_H

