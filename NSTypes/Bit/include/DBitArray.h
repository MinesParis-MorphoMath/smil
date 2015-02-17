/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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


#include "Core/include/private/DTypes.hpp"

#include <iostream>
#include <algorithm>



namespace smil
{
    typedef bool Bit;
    
    class BitArray
    {
    public:


        typedef size_t INT_TYPE;

        UINT index;
        INT_TYPE *intArray;
        
        
        static const INT_TYPE INT_TYPE_SIZE = sizeof(INT_TYPE)*CHAR_BIT;
        static inline INT_TYPE INT_TYPE_MIN() { return numeric_limits<INT_TYPE>::min(); }
        static inline INT_TYPE INT_TYPE_MAX() { return numeric_limits<INT_TYPE>::max(); }
        static inline UINT INT_SIZE(UINT bitCount) { return (bitCount-1)/INT_TYPE_SIZE + 1; }
        
        //! Most significant bit
        static const INT_TYPE INT_MS_BIT = (1UL << (INT_TYPE_SIZE - 2));
        //! Less significant bit
        static const INT_TYPE INT_LS_BIT = 0x01;
        
        
        BitArray();
        BitArray(const BitArray &rhs);
        BitArray(UINT _bitWidth, UINT _bitHeight);
        BitArray(INT_TYPE *arr, UINT _bitWidth, UINT _bitHeight=1);
        BitArray(bool *arr, UINT _bitWidth, UINT _bitHeight=1);
        ~BitArray();
        
        
        inline UINT getBitWidth() { return bitWidth; }
        inline UINT getIntWidth() { return intWidth; }
        inline UINT getIntNbr() { return intWidth*height; }
        inline UINT getHeight() { return height; }
        inline UINT getBitPadX() const { return intWidth*INT_TYPE_SIZE - bitWidth; }

        
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
        
        inline bool getValue() const
        {
            return getValue(index);
        }
        inline bool getValue(UINT ind) const
        {
            int Y = ind / bitWidth;
            int X = (ind + Y*this->getBitPadX()) / INT_TYPE_SIZE;
            int x = (ind-Y*bitWidth) % INT_TYPE_SIZE;
            return (intArray[X] & (1UL << x))!=0;
        }
        inline void setValue(bool v) 
        {
            setValue(index, v);
        }
        inline void setValue(UINT ind, bool val)
        {
            int Y = ind / bitWidth;
            int X = (ind + Y*this->getBitPadX()) / INT_TYPE_SIZE;
            int x = (ind-Y*bitWidth) % INT_TYPE_SIZE;
            if (val)
                intArray[X] |= (1UL << x);
            else intArray[X] &= ~(1UL << x);
        }
        
        operator bool() { return getValue(index); }
        BitArray operator [] (UINT i); // lValue
        const bool operator [] (UINT i) const; // rValue
        
        inline BitArray &operator * ()
        {
            return *this;
        }
        
    //     operator void* () { return (void*)this->intArray; }
    //     operator char* () { return (char*)this->intArray; }
//         void operator=(void *ptr) { this->intArray = (INT_TYPE*)ptr; }
        const bool &operator=(const bool &b) 
        { 
            setValue(b);
            return b; 
        }
        inline BitArray operator + (int dp) const;
        inline BitArray operator + (long unsigned int dp) const
        {
            return operator+((int)dp);
        }
        inline BitArray operator + (UINT dp) const
        {
            return operator+((int)dp);
        }
        BitArray operator - (int dp) const;
        BitArray& operator ++ (int);
        BitArray& operator ++ ();
        
        BitArray& operator += (int dp) { index+=dp; return *this; }
        
        inline BitArray& operator = (const BitArray &rhs)
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
    

    BitArray BitArray::operator + (int dp) const
    {
        BitArray ba(this->intArray, this->bitWidth, this->height);
        ba.index = this->index + dp;
        return ba;
    }

    
    
    
    

    inline ostream& operator << (ostream &os, BitArray &b)
    {
        return b.printSelf(os);
    }


    template <>
    inline const char *getDataTypeAsString(Bit *val)
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
        static inline size_t cardinal() { return 2; }
        static inline lineType createLine(UINT lineLen) 
        { 
            BitArray ba(lineLen, 1);
            ba.createIntArray();
            return ba; 
        }
        static inline void deleteLine(lineType line) 
        { 
            line.deleteIntArray();
        }
        static inline unsigned long ptrOffset(lineType p, unsigned long n=SIMD_VEC_SIZE) { return ((unsigned long)(p.intArray)) & (n-1); }
        static inline std::string toString(const Bit &val)
        {
            stringstream str;
            str << int(val);
            return str.str();
        }
    };
} // namespace smil


#endif // _DBITARRAY_H

