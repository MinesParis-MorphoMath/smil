/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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

using namespace smil;


BitArray::BitArray()
  : index(0), intArray(NULL), intWidth(0), bitWidth(0), height(0)
{
}

BitArray::BitArray(const BitArray &rhs)
{
    this->setSize(rhs.bitWidth, rhs.height);
    this->intArray = rhs.intArray;
    this->index = rhs.index;
}

BitArray::BitArray(UINT _bitWidth, UINT _bitHeight)
  : index(0), intArray(NULL)
{
    setSize(_bitWidth, _bitHeight);
}
BitArray::BitArray(INT_TYPE *arr, UINT _bitWidth, UINT _bitHeight)
  : index(0), intArray(arr)
{
    setSize(_bitWidth, _bitHeight);
}
BitArray::BitArray(bool *arr, UINT _bitWidth, UINT _bitHeight)
  : index(0), intArray(NULL)
{
    setSize(_bitWidth, _bitHeight);
    createIntArray();
    for (size_t i=0;i<_bitWidth;i++)
      setValue(i, arr[i]);
}

BitArray::~BitArray()
{
    intArray = NULL;
}

void BitArray::setSize(UINT _bitWidth, UINT _bitHeight)
{
    bitWidth = _bitWidth;
    intWidth = INT_SIZE(bitWidth);
    height = _bitHeight;
}


BitArray BitArray::operator [] (UINT i)
{
    BitArray ba;
    ba.setSize(bitWidth, height);
    ba.intArray = intArray;
    ba.index = this->index + i;
    return ba;
}

const bool BitArray::operator [] (UINT i) const
{
    return Bit(this->getValue(i));
}



BitArray BitArray::operator-(int dp) const
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


