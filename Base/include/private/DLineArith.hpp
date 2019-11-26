/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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


#ifndef _D_LINE_ARITH_HPP
#define _D_LINE_ARITH_HPP


#include "DBaseLineOperations.hpp"


namespace smil
{
  
    /**
    * @ingroup Arith
    * @{
    */

    /**
    * copy line
    */
    template <class T1, class T2>
    inline void copyLine(const typename unaryLineFunctionBase<T1>::lineInType lIn, const size_t size, typename unaryLineFunctionBase<T2>::lineInType lOut)
    {
        for (size_t i=0;i<size;i++)
          lOut[i] = lIn[i];
    }

    template <class T>
    inline void copyLine(const typename unaryLineFunctionBase<T>::lineInType lIn, const size_t size, typename unaryLineFunctionBase<T>::lineInType lOut)
    {
        for (size_t i=0;i<size;i++)
          lOut[i] = lIn[i];
    }


    template <class T>
    struct fillLine : public unaryLineFunctionBase<T>
    {
        typedef typename unaryLineFunctionBase<T>::lineType lineType;
        fillLine() {}
        fillLine(const lineType lIn, const size_t size, const T value) { this->_exec(lIn, size, value); }
        
        virtual void _exec(const lineType lIn, const size_t size, lineType lOut)
        {
            return this->fill(lIn, size, lOut);
        }
        virtual void _exec(lineType lInOut, const size_t size, const T value)
        {
            return this->fill(lInOut, size, value);
        }
        static void fill(const lineType lIn, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
              lOut[i] = lIn[i];
        }
        static void fill(lineType lInOut, const size_t size, const T value)
        {
            for (size_t i=0;i<size;i++)
              lInOut[i] = value;
        }
    };

    template <class T>
    inline void shiftLine(const typename unaryLineFunctionBase<T>::lineInType lIn, int dx, size_t lineLen, typename unaryLineFunctionBase<T>::lineInType lOut, T borderValue = ImDtTypes<T>::min())
    {
        if (dx==0)
            copyLine<T>(lIn, lineLen, lOut);
        else if (dx>0)
        {
            fillLine<T>::fill(lOut, dx, borderValue);
            typename Image<T>::restrictLineType tmpL = lOut+dx;
            copyLine<T>(lIn, lineLen-dx, tmpL);
        }
        else
        {
            typename Image<T>::restrictLineType tmpL = lIn-dx;
            copyLine<T>(tmpL, lineLen+dx, lOut);
            fillLine<T>::fill(lOut+(lineLen+dx), -dx, borderValue);
        }
    }

    template <class T>
    struct invLine : public unaryLineFunctionBase<T>
    {
        typedef typename unaryLineFunctionBase<T>::lineType lineType;
        inline void _exec(const lineType lineIn, const size_t size, lineType lOut)
        {

            for (size_t i=0;i<size;i++)
              lOut[i] = ImDtTypes<T>::max() - lineIn[i] - ImDtTypes<T>::min();
        }
    };

    template <class T>
    struct addLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] > (T)(ImDtTypes<T>::max()- lIn2[i]) ? ImDtTypes<T>::max() : lIn1[i] + lIn2[i];
        }
    };

    template <class T>
    struct addNoSatLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] + lIn2[i];
        }
    };

    template <class T>
    struct subLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] < (T)(ImDtTypes<T>::min() + lIn2[i]) ? ImDtTypes<T>::min() : lIn1[i] - lIn2[i];
        }
    };

    template <class T>
    struct subNoSatLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] - lIn2[i];
        }
    };

    template <class T>
    struct supLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] > lIn2[i] ? lIn1[i] : lIn2[i];
        }
    };

    template <class T>
    struct infLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] < lIn2[i] ? lIn1[i] : lIn2[i];
        }
    };

    template <class T>
    struct grtLine : public binaryLineFunctionBase<T>
    {
        grtLine() 
          : trueVal(ImDtTypes<T>::max()), falseVal(0) {}
          
        T trueVal, falseVal;
          
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            T _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] > lIn2[i] ? _trueVal : _falseVal;
        }
    };

    template <class T_in, class T_out=T_in>
    struct grtSupLine : public binaryLineFunctionBase<T_in, T_in, T_out>
    {
        grtSupLine() 
          : trueVal(ImDtTypes<T_out>::max()), falseVal(0) {}
          
        T_out trueVal, falseVal;
          
        typedef binaryLineFunctionBase<T_in,T_in,T_out> parentClass;
        typedef typename parentClass::lineType lineInType;
        typedef typename parentClass::lineOutType lineOutType;
        
        virtual void _exec(const lineInType lIn1, const lineInType lIn2, const size_t size, lineOutType lOut)
        {
            T_out _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] |= lIn1[i] > lIn2[i] ? _trueVal : _falseVal;
        }
    };

    template <class T>
    struct grtOrEquLine : public binaryLineFunctionBase<T>
    {
        grtOrEquLine() 
          : trueVal(ImDtTypes<T>::max()), falseVal(0) {}
          
        T trueVal, falseVal;
          
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            T _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] >= lIn2[i] ? _trueVal : _falseVal;
        }
    };

    template <class T_in, class T_out=T_in>
    struct grtOrEquSupLine : public binaryLineFunctionBase<T_in, T_in, T_out>
    {
        grtOrEquSupLine() 
          : trueVal(ImDtTypes<T_out>::max()), falseVal(0) {}
          
        T_out trueVal, falseVal;
          
        typedef binaryLineFunctionBase<T_in,T_in,T_out> parentClass;
        typedef typename parentClass::lineType lineInType;
        typedef typename parentClass::lineOutType lineOutType;
        
        virtual void _exec(const lineInType lIn1, const lineInType lIn2, const size_t size, lineOutType lOut)
        {
            T_out _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] |= lIn1[i] >= lIn2[i] ? _trueVal : _falseVal;
        }
    };

    template <class T>
    struct lowLine : public binaryLineFunctionBase<T>
    {
        lowLine() 
          : trueVal(ImDtTypes<T>::max()), falseVal(0) {}
          
        T trueVal, falseVal;
          
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            T _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] < lIn2[i] ? _trueVal : _falseVal;
        }
    };

    template <class T_in, class T_out=T_in>
    struct lowSupLine : public binaryLineFunctionBase<T_in, T_in, T_out>
    {
        lowSupLine() 
          : trueVal(ImDtTypes<T_out>::max()), falseVal(0) {}
          
        T_out trueVal, falseVal;
          
        typedef binaryLineFunctionBase<T_in,T_in,T_out> parentClass;
        typedef typename parentClass::lineType lineInType;
        typedef typename parentClass::lineOutType lineOutType;
        
        virtual void _exec(const lineInType lIn1, const lineInType lIn2, const size_t size, lineOutType lOut)
        {
            T_out _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] |= lIn1[i] < lIn2[i] ? _trueVal : _falseVal;
        }
    };

    template <class T>
    struct lowOrEquLine : public binaryLineFunctionBase<T>
    {
        lowOrEquLine() 
          : trueVal(ImDtTypes<T>::max()), falseVal(0) {}
          
        T trueVal, falseVal;
          
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            T _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] <= lIn2[i] ? _trueVal : _falseVal;
        }
    };

    template <class T_in, class T_out=T_in>
    struct lowOrEquSupLine : public binaryLineFunctionBase<T_in, T_in, T_out>
    {
        lowOrEquSupLine() 
          : trueVal(ImDtTypes<T_out>::max()), falseVal(0) {}
          
        T_out trueVal, falseVal;
          
        typedef binaryLineFunctionBase<T_in,T_in,T_out> parentClass;
        typedef typename parentClass::lineType lineInType;
        typedef typename parentClass::lineOutType lineOutType;
        
        virtual void _exec(const lineInType lIn1, const lineInType lIn2, const size_t size, lineOutType lOut)
        {
            T_out _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] |= lIn1[i] <= lIn2[i] ? _trueVal : _falseVal;
        }
    };

    
    template <class T>
    struct equLine : public binaryLineFunctionBase<T>
    {
        equLine() 
          : trueVal(ImDtTypes<T>::max()), falseVal(0) {}
          
        T trueVal, falseVal;
          
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            T _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] == lIn2[i] ? _trueVal : _falseVal;
        }
    };

    template <class T>
    struct diffLine : public binaryLineFunctionBase<T>
    {
        diffLine() 
          : trueVal(ImDtTypes<T>::max()), falseVal(0) {}
          
        T trueVal, falseVal;
          
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            T _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] = (lIn1[i] == lIn2[i]) ? _falseVal : _trueVal;
        }
    };

    template <class T_in, class T_out=T_in>
    struct equSupLine : public binaryLineFunctionBase<T_in, T_in, T_out>
    {
        equSupLine() 
          : trueVal(ImDtTypes<T_out>::max()), falseVal(0) {}
          
        T_out trueVal, falseVal;
          
        typedef binaryLineFunctionBase<T_in,T_in,T_out> parentClass;
        typedef typename parentClass::lineType lineInType;
        typedef typename parentClass::lineOutType lineOutType;
        
        virtual void _exec(const lineInType lIn1, const lineInType lIn2, const size_t size, lineOutType lOut)
        {
            T_out _trueVal(trueVal), _falseVal(falseVal);
            for (size_t i=0;i<size;i++)
                lOut[i] |= lIn1[i] == lIn2[i] ? _trueVal : _falseVal;
        }
    };


    /*
    * Difference ("vertical distance") between two lines.
    * 
    * Returns abs(p1-p2) for each pixels pair
    */

    template <class T>
    struct absDiffLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = lIn1[i] > lIn2[i] ? lIn1[i]-lIn2[i] : lIn2[i]-lIn1[i];
        }
    };


    template <class T>
    struct mulLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = double(lIn1[i]) * double(lIn2[i]) > double(ImDtTypes<T>::max()) ? ImDtTypes<T>::max() : lIn1[i] * lIn2[i];
        }
    };

    template <class T>
    struct mulNoSatLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = (T)(lIn1[i] * lIn2[i]);
        }
    };

    template <class T>
    struct divLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
            {
                lOut[i] = lIn2[i]==T(0) ? ImDtTypes<T>::max() : lIn1[i] / lIn2[i];
            }
        }
    };

    template <class T>
    struct logLine : public unaryLineFunctionBase<T>
    {
        typedef typename unaryLineFunctionBase<T>::lineType lineType;
        int base;
        logLine() : base(0) {}
        virtual void _exec(const lineType lIn, const size_t size, lineType lOut)
        {
            if (base!=0)
            {
                double baseLog = std::log(double(base));
                for (size_t i=0;i<size;i++)
                    lOut[i] = std::log(lIn[i]) / baseLog;
            }
            else
            {
              for (size_t i=0;i<size;i++)
                  lOut[i] = std::log(lIn[i]);
            }
        }
    };

    template <class T>
    struct logicAndLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = (T)(lIn1[i] && lIn2[i]);
        }
    };

    template <class T>
    struct bitAndLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = (T)(lIn1[i] & lIn2[i]);
        }
    };

    template <class T>
    struct logicOrLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = (T)(lIn1[i] || lIn2[i]);
        }
    };

    template <class T>
    struct bitOrLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = (T)(lIn1[i] | lIn2[i]);
        }
    };
    
    template <class T>
    struct logicXOrLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = (T)((lIn1[i] && !lIn2[i]) || (!lIn1[i] && lIn2[i]));
        }
    };

    template <class T>
    struct bitXOrLine : public binaryLineFunctionBase<T>
    {
        typedef typename binaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType lIn1, const lineType lIn2, const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
                lOut[i] = (T)(lIn1[i] ^ lIn2[i]);
        }
    };

    template <class T>
    struct leftShiftLine : public unaryLineFunctionBase<T>
    {
        typedef typename unaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType, const size_t, lineType) {}
        virtual void _exec(const lineType lIn1, const UINT shift , const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
              lOut[i] = (T)(lIn1[i] << shift);
        }
    };

    template <class T>
    struct rightShiftLine : public unaryLineFunctionBase<T>
    {
        using unaryLineFunctionBase<T>::_exec;
        typedef typename unaryLineFunctionBase<T>::lineType lineType;
        virtual void _exec(const lineType, const size_t, lineType) {}
        virtual void _exec(const lineType lIn1, const UINT shift , const size_t size, lineType lOut)
        {
            for (size_t i=0;i<size;i++)
              lOut[i] = (T)(lIn1[i] >> shift);
        }
    };

    template <class T1, class T2>
    struct testLine : public tertiaryLineFunctionBase<T1,T2,T2,T2>
    {
        typedef tertiaryLineFunctionBase<T1,T2,T2,T2> parentClass;
        typedef typename parentClass::lineType1 lineType1;
        typedef typename parentClass::lineOutType lineType2;
        
        virtual void _exec(const lineType1 lIn1, const lineType2 lIn2, const lineType2 lIn3, const size_t size, lineType2 lOut)
        {
            for (size_t i=0;i<size;i++)
            {
                lOut[i] = lIn1[i] ? lIn2[i] : lIn3[i];
            }
        }
    };

/** @}*/

} // namespace smil


#endif // _D_LINE_ARITH_HPP
