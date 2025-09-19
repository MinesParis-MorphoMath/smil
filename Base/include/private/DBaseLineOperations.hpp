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


#ifndef _BASE_LINE_OPERATIONS_HPP
#define _BASE_LINE_OPERATIONS_HPP


#include "Core/include/private/DImage.hpp"


namespace smil
{
  
    template <class T> class Image;


    // Base abstract struct of line unary function
    template <class T, class T_out=T>
    struct unaryLineFunctionBase
    {
        typedef Image<T> imageInType;
        typedef typename imageInType::restrictLineType lineInType;
        typedef typename imageInType::sliceType sliceInType;
        
        typedef Image<T_out> imageOutType;
        typedef typename imageOutType::restrictLineType lineOutType;
        typedef typename imageOutType::sliceType sliceOutType;
        
        typedef lineInType lineType;
        
        unaryLineFunctionBase() {}
        unaryLineFunctionBase(const lineInType lineIn, const size_t size, lineOutType lineOut)
        {
            this->_exec(lineIn, size, lineOut);
        }
        virtual ~unaryLineFunctionBase() {}
        
        virtual void _exec(const lineInType, const size_t, lineOutType) = 0;
        virtual void _exec_aligned(const lineInType lineIn, const size_t size, lineOutType lineOut) { _exec(lineIn, size, lineOut); }
        virtual void _exec(lineOutType, const size_t, const T_out) {}
        virtual void _exec_aligned(lineOutType lineOut, const size_t size, T_out value) { _exec(lineOut, size, value); }
        inline void operator()(const lineInType lineIn, const size_t size, lineOutType lineOut)
        { 
            unsigned long ptrOffset1 = ImDtTypes<T>::ptrOffset(lineIn);
            unsigned long ptrOffset2 = ImDtTypes<T_out>::ptrOffset(lineOut);
            
            // both aligned
            if (!ptrOffset1 && !ptrOffset2)
            {
                _exec_aligned(lineIn, size, lineOut);
            }
            // both misaligned but with same misalignment
            else if (ptrOffset1==ptrOffset2)
            {
                unsigned long misAlignSize = SIMD_VEC_SIZE - ptrOffset1;
                _exec(lineIn, misAlignSize, lineOut); 
                _exec_aligned(lineIn+misAlignSize, size-misAlignSize, lineOut+misAlignSize); 
            }
            // both misaligned with different misalignments
            else
            {
                _exec(lineIn, size, lineOut); 
            }
        }
        inline void operator()(const lineInType lineIn, const size_t size, T_out value)
        { 
            if (size<SIMD_VEC_SIZE)
            {
                _exec(lineIn, size, value);
                return;
            }
            size_t ptrOffset = ImDtTypes<T>::ptrOffset(lineIn);
            size_t misAlignSize = ptrOffset==0 ? 0 : SIMD_VEC_SIZE - ptrOffset;
            if (misAlignSize)
              _exec(lineIn, misAlignSize, value); 
            _exec_aligned(lineIn+misAlignSize, size-misAlignSize, value); 
        }
    };


    // Base abstract struct of line binary function
    template <class T1, class T2=T1, class T_out=T1>
    struct binaryLineFunctionBase
    {
        virtual ~binaryLineFunctionBase() {}
        typedef typename Image<T1>::restrictLineType lineType1;
        typedef typename Image<T2>::restrictLineType lineType2;
        typedef typename Image<T_out>::restrictLineType lineOutType;
        typedef lineType1 lineType;
        
        typedef typename Image<T1>::sliceType sliceType;
        
        virtual void _exec(const lineType1, const lineType2, const size_t, lineOutType) = 0;
        virtual void _exec_aligned(const lineType1 lineIn1, const lineType2 lineIn2, const size_t size, lineOutType lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
        inline void operator()(const lineType1 lineIn1, const lineType2 lineIn2, const size_t size, lineOutType lineOut)
        { 
            return _exec(lineIn1, lineIn2, size, lineOut); 
            if (size<SIMD_VEC_SIZE)
            {
                _exec(lineIn1, lineIn2, size, lineOut); 
                return;
            }
            size_t ptrOffset1 = ImDtTypes<T1>::ptrOffset(lineIn1);
            size_t ptrOffset2 = ImDtTypes<T2>::ptrOffset(lineIn2);
            size_t ptrOffset3 = ImDtTypes<T_out>::ptrOffset(lineOut);
            
            // all aligned
            if (!ptrOffset1 && !ptrOffset2 && !ptrOffset3)
            {
                _exec_aligned(lineIn1, lineIn2, size, lineOut);
            }
            // all misaligned but with same misalignment
            else if (ptrOffset1==ptrOffset2 && ptrOffset2==ptrOffset3)
            {
                size_t misAlignSize = SIMD_VEC_SIZE - ptrOffset1;
                _exec(lineIn1, lineIn2, misAlignSize, lineOut); 
                _exec_aligned(lineIn1+misAlignSize, lineIn2+misAlignSize, size-misAlignSize, lineOut+misAlignSize); 
            }
            // all misaligned with different misalignments
            else 
            {
                _exec(lineIn1, lineIn2, size, lineOut); 
            }
            
        }
        inline void operator()(const lineType1 lineIn1, const T2 value, const size_t size, lineOutType lineOut)
        { 
            if (size<SIMD_VEC_SIZE)
            {
                _exec(lineIn1, value, size, lineOut); 
                return;
            }
            unsigned long ptrOffset1 = ImDtTypes<T1>::ptrOffset(lineIn1);
            unsigned long ptrOffset2 = ImDtTypes<T2>::ptrOffset(lineOut);
            
            // all aligned
            if (!ptrOffset1 && !ptrOffset2)
            {
                _exec_aligned(lineIn1, value, size, lineOut);
            }
            // all misaligned but with same misalignment
            else if (ptrOffset1==ptrOffset2)
            {
                unsigned long misAlignSize = SIMD_VEC_SIZE - ptrOffset1;
                _exec(lineIn1, value, misAlignSize, lineOut); 
                _exec_aligned(lineIn1+misAlignSize, value, size-misAlignSize, lineOut+misAlignSize); 
            }
            // all misaligned with different misalignments
            else 
            {
                _exec(lineIn1, value, size, lineOut); 
            }
            
        }
    };


    // Base abstract struct of line binary function
    template <class T1, class T2=T1, class T3=T1, class T_out=T1>
    struct tertiaryLineFunctionBase
    {
        virtual ~tertiaryLineFunctionBase() {}
        typedef typename Image<T1>::restrictLineType lineType1;
        typedef typename Image<T2>::restrictLineType lineType2;
        typedef typename Image<T3>::restrictLineType lineType3;
        typedef typename Image<T_out>::restrictLineType lineOutType;
        typedef lineType1 lineType;
        
        virtual void _exec(const lineType1 /*lineIn1*/, const lineType2 /*lineIn2*/, const lineType3 /*lineIn3*/, const size_t /*size*/, lineOutType /*lineOut*/) = 0;
        virtual void _exec_aligned(const lineType1 lineIn1, const lineType2 lineIn2, const lineType3 lineIn3, const size_t size, lineOutType lineOut) { _exec(lineIn1, lineIn2, lineIn3, size, lineOut); }
        virtual void operator()(const lineType1 lineIn1, const lineType2 lineIn2, const lineType3 lineIn3, const size_t size, lineOutType lineOut)
        { 
            if (size<SIMD_VEC_SIZE)
            {
                _exec(lineIn1, lineIn2, lineIn3, size, lineOut);
                return;
            }
            unsigned long ptrOffset1 = ImDtTypes<T1>::ptrOffset(lineIn1);
            unsigned long ptrOffset2 = ImDtTypes<T2>::ptrOffset(lineIn2);
            unsigned long ptrOffset3 = ImDtTypes<T3>::ptrOffset(lineIn3);
            unsigned long ptrOffset4 = ImDtTypes<T_out>::ptrOffset(lineOut);
            
            // all aligned
            if (!ptrOffset1 && !ptrOffset2 && !ptrOffset3 && !ptrOffset4)
            {
                _exec_aligned(lineIn1, lineIn2, lineIn3, size, lineOut);
            }
            // all misaligned but with same misalignment
            else if (ptrOffset1==ptrOffset2 && ptrOffset2==ptrOffset3)
            {
                size_t misAlignSize = SIMD_VEC_SIZE - ptrOffset1;
                _exec(lineIn1, lineIn2, lineIn3, misAlignSize, lineOut); 
                _exec_aligned(lineIn1+misAlignSize, lineIn2+misAlignSize, lineIn3+misAlignSize, size-misAlignSize, lineOut+misAlignSize); 
            }
            // all misaligned with different misalignments
            else 
            {
                _exec(lineIn1, lineIn2, lineIn3, size, lineOut); 
            }
        }
    };

} // namespace smil




#endif
