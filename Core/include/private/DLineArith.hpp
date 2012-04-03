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


#ifndef _D_LINE_ARITH_HPP
#define _D_LINE_ARITH_HPP


#include "DImage.h"


/**
 * \ingroup Core
 * \defgroup Arith
 * @{
 */

template <class T1, class T2>
inline void copyLine(typename Image<T1>::lineType lIn, int size, typename Image<T2>::lineType lOut)
{
    for (int i=0;i<size;i++)
      lOut[i] = static_cast<T2>(lIn[i]);
}

template <class T>
inline void copyLine(typename Image<T>::lineType &lIn, int size, typename Image<T>::lineType &lOut)
{
    memcpy(lOut, lIn, size*sizeof(T));
}



template <class T>
struct fillLine : public unaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    fillLine() {}
    fillLine(lineType lIn, int size, T value) { this->_exec(lIn, size, value); }
    
    inline void _exec(lineType lIn, int size, lineType lOut)
    {
	memcpy(lOut, lIn, size*sizeof(T));
    }
    inline void _exec(lineType lInOut, int size, T value)
    {
        for (int i=0;i<size;i++)
            lInOut[i] = value;
    }
};

template <class T>
inline void shiftLine(typename Image<T>::lineType &lIn, int dx, int lineLen, typename Image<T>::lineType &lOut, T borderValue = numeric_limits<T>::min())
{
    fillLine<T> fillFunc;

    if (dx==0)
        copyLine<T>(lIn, lineLen, lOut);
    else if (dx>0)
    {
        fillFunc(lOut, dx, borderValue);
	typename Image<T>::lineType tmpL = lOut+dx;
        copyLine<T>(lIn, lineLen-dx, tmpL);
    }
    else
    {
	typename Image<T>::lineType tmpL = lIn-dx;
        copyLine<T>(tmpL, lineLen+dx, lOut);
        fillFunc(lOut+(lineLen+dx), -dx, borderValue);
    }
}

template <class T>
struct invLine : public unaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lineIn, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = ~lineIn[i];
    }
};

template <class T>
struct addLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] > (T)(numeric_limits<T>::max()- lIn2[i]) ? numeric_limits<T>::max() : lIn1[i] + lIn2[i];
    }
};

template <class T>
struct addNoSatLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] + lIn2[i];
    }
};

template <class T>
struct subLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] < (T)(numeric_limits<T>::max() + lIn2[i]) ? numeric_limits<T>::min() : lIn1[i] - lIn2[i];
    }
};

template <class T>
struct subNoSatLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] - lIn2[i];
    }
};

template <class T>
struct supLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] > lIn2[i] ? lIn1[i] : lIn2[i];
    }
};

template <class T>
struct infLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] < lIn2[i] ? lIn1[i] : lIn2[i];
    }
};

template <class T>
struct grtLine : public binaryLineFunctionBase<T>
{
    grtLine() 
      : trueVal(numeric_limits<T>::max()), falseVal(0) {}
      
    T trueVal, falseVal;
      
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] > lIn2[i] ? trueVal : falseVal;
    }
};

template <class T>
struct grtOrEquLine : public binaryLineFunctionBase<T>
{
    grtOrEquLine() 
      : trueVal(numeric_limits<T>::max()), falseVal(0) {}
      
    T trueVal, falseVal;
      
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] >= lIn2[i] ? trueVal : falseVal;
    }
};

template <class T>
struct lowLine : public binaryLineFunctionBase<T>
{
    lowLine() 
      : trueVal(numeric_limits<T>::max()), falseVal(0) {}
      
    T trueVal, falseVal;
      
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] < lIn2[i] ? trueVal : falseVal;
    }
};

template <class T>
struct lowOrEquLine : public binaryLineFunctionBase<T>
{
    lowOrEquLine() 
      : trueVal(numeric_limits<T>::max()), falseVal(0) {}
      
    T trueVal, falseVal;
      
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] <= lIn2[i] ? trueVal : falseVal;
    }
};

template <class T>
struct equLine : public binaryLineFunctionBase<T>
{
    equLine() 
      : trueVal(numeric_limits<T>::max()), falseVal(0) {}
      
    T trueVal, falseVal;
      
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] == lIn2[i] ? trueVal : falseVal;
    }
};


/**
 * Difference ("vertical distance") between two lines.
 * 
 * Returns abs(p1-p2) for each pixels pair
 */

template <class T>
struct diffLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = lIn1[i] > lIn2[i] ? lIn1[i]-lIn2[i] : lIn2[i]-lIn1[i];
    }
};


template <class T>
struct mulLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = double(lIn1[i]) * double(lIn2[i]) > double(numeric_limits<T>::max()) ? numeric_limits<T>::max() : lIn1[i] * lIn2[i];
    }
};

template <class T>
struct mulNoSatLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = (T)(lIn1[i] * lIn2[i]);
    }
};

template <class T>
struct divLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
        {
            lOut[i] = lIn2[i]==0 ? numeric_limits<T>::max() : lIn1[i] / lIn2[i];
        }
    }
};

template <class T>
struct logicAndLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = (T)(lIn1[i] && lIn2[i]);
    }
};

template <class T>
struct logicOrLine : public binaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
            lOut[i] = (T)(lIn1[i] || lIn2[i]);
    }
};


template <class T>
struct testLine : public tertiaryLineFunctionBase<T>
{
    typedef typename Image<T>::lineType lineType;
    inline void _exec(lineType lIn1, lineType lIn2, lineType lIn3, int size, lineType lOut)
    {
        for (int i=0;i<size;i++)
        {
            lOut[i] = lIn1[i] ? lIn2[i] : lIn3[i];
        }
    }
};


/** @}*/

#endif // _D_LINE_ARITH_HPP
