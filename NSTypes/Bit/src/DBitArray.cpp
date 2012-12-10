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


#include "DBitArray.h"

// const BitArray::INT_TYPE BitArray::INT_TYPE_SIZE = sizeof(INT_TYPE)*CHAR_BIT;


void BitArray::setSize(UINT _bitWidth, UINT _bitHeight)
{
    bitWidth = _bitWidth;
    intWidth = INT_SIZE(bitWidth);
    height = _bitHeight;
}

bool BitArray::getValue(UINT ind)
{
    int Y = ind / bitWidth;
    int X = (ind + Y*this->getBitPadX()) / INT_TYPE_SIZE;
    int x = (ind-Y*bitWidth) % INT_TYPE_SIZE;
    return (intArray[X] & (1UL << x))!=0;
}

void BitArray::setValue(UINT ind, bool val)
{
    int Y = ind / bitWidth;
    int X = (ind + Y*this->getBitPadX()) / INT_TYPE_SIZE;
    int x = (ind-Y*bitWidth) % INT_TYPE_SIZE;
    if (val)
        intArray[X] |= (1UL << x);
    else intArray[X] &= ~(1UL << x);
}

Bit BitArray::operator [] (UINT i)
{
    Bit b;
    b.bitArray = this;
    b.index = i;
    return b;
}

Bit BitArray::operator [] (UINT i) const
{
    Bit b;
    b.bitArray = (BitArray*)this;
    b.index = i;
    return b;
}



BitArray BitArray::operator-(int dp)
{
    BitArray ba(this->intArray, this->bitWidth, this->height);
    ba.index = this->index - dp;
    return ba;
}

BitArray& BitArray::operator++(int)
{
    index++;
    return *this;
}

BitArray& BitArray::operator++()
{
    index++;
    return *this;
}


ostream& BitArray::printSelf(ostream &os)
{
    if (!this->intArray)
      return os;
    
    for (UINT i=0;i<bitWidth;i++)
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



Bit::operator bool() const
{
    if (bitArray)
	return bitArray->getValue(index);
    else return value;
}

Bit& Bit::operator = (const bool v)
{
    if (bitArray)
        bitArray->setValue(index, v);
    else value = v;
    return *this;
}

Bit& Bit::operator = (const Bit &src)
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

