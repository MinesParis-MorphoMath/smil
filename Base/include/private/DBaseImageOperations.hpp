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


#ifndef _BASE_IMAGE_OPERATIONS_HPP
#define _BASE_IMAGE_OPERATIONS_HPP

#include "Core/include/private/DImage.hpp"
#include "Core/include/private/DMemory.hpp"

#include "DBaseLineOperations.hpp"

namespace smil
{
 
    template <class T>
    class imageFunctionBase
    {
    public:
        typedef Image<T> imageType;
        typedef typename imageType::sliceType sliceType;
        typedef typename imageType::lineType lineType;
        typedef typename imageType::pixelType pixelType;

        sliceType alignedBuffers;
        size_t bufferNumber;
        size_t bufferLength;
        size_t bufferSize;
        
        RES_T retVal;

        imageFunctionBase() : alignedBuffers ( NULL ) {};
        ~imageFunctionBase()
        {
            deleteAlignedBuffers();
        };

        lineType *createAlignedBuffers ( size_t nbr, size_t len );
        void deleteAlignedBuffers();
        inline void copyLineToBuffer ( T *line, size_t bufIndex );
        inline void copyBufferToLine ( size_t bufIndex, T *line );
        
        operator RES_T() { return retVal; }
    };

    template <class T, class lineFunction_T, class T_out=T>
    class unaryImageFunction : public imageFunctionBase<T>
    {
    public:
        typedef imageFunctionBase<T> parentClass;
        typedef Image<T> imageInType;
        typedef typename imageInType::pixelType pixelInType;
        typedef typename imageInType::lineType lineInType;
        typedef typename imageInType::sliceType sliceInType;

        typedef Image<T_out> imageOutType;
        typedef typename imageOutType::pixelType pixelOutType;
        typedef typename imageOutType::lineType lineOutType;
        typedef typename imageOutType::sliceType sliceOutType;

        unaryImageFunction() {}
        unaryImageFunction( const imageInType &imIn, imageOutType &ImOut ) 
        {
            this->retVal = this->_exec ( imIn, ImOut );
        }
        unaryImageFunction( imageOutType &imOut, const T_out &value ) 
        {
            this->retVal = this->_exec (imOut, value);
        }
        
        inline RES_T operator() ( const imageInType &imIn, imageOutType &ImOut )
        {
            return this->_exec ( imIn, ImOut );
        }
        inline RES_T operator() ( imageOutType &ImOut, const T_out &value )
        {
            return this->_exec ( ImOut, value );
        }

        RES_T _exec ( const imageInType &imIn, imageOutType &imOut );
        RES_T _exec ( imageOutType &imOut, const T_out &value );

    //   protected:
        lineFunction_T lineFunction;
    };


    template <class T, class lineFunction_T>
    class binaryImageFunction : public imageFunctionBase<T>
    {
    public:
        typedef imageFunctionBase<T> parentClass;
        typedef Image<T> imageType;
        typedef typename imageType::pixelType pixelType;
        typedef typename imageType::lineType lineType;
        typedef typename imageType::sliceType sliceType;

        binaryImageFunction() {}
        binaryImageFunction ( const imageType &imIn1, const imageType &imIn2, imageType &ImOut )
        {
            this->retVal = this->_exec ( imIn1, imIn2, ImOut );
        }
        binaryImageFunction ( const imageType &imIn, const T &value, imageType &ImOut )
        {
            this->retVal = this->_exec ( imIn, value, ImOut );
        }
        binaryImageFunction ( const T &value, const imageType &imIn, imageType &ImOut )
        {
            this->retVal = this->_exec ( value, imIn, ImOut );
        }
        
        inline RES_T operator() ( const imageType &imIn1, const imageType &imIn2, imageType &ImOut )
        {
            return this->_exec ( imIn1, imIn2, ImOut );
        }
        inline RES_T operator() ( const imageType &imIn, const T &value, imageType &ImOut )
        {
            return this->_exec ( imIn, value, ImOut );
        }

        inline RES_T operator() (const T &value, const imageType &imIn, imageType &ImOut )
        {
            return this->_exec ( value, imIn, ImOut );
        }

        RES_T _exec ( const imageType &imIn1, const imageType &imIn2, imageType &imOut );
        RES_T _exec ( const imageType &imIn, imageType &imInOut );
        RES_T _exec ( const imageType &imIn, const T &value, imageType &imOut );
        RES_T _exec ( const T &value, const imageType &imIn, imageType &imOut );

    //   protected:
        lineFunction_T lineFunction;
    };

    template <class T, class lineFunction_T>
    class tertiaryImageFunction : public imageFunctionBase<T>
    {
    public:
        typedef imageFunctionBase<T> parentClass;
        typedef Image<T> imageType;
        typedef typename imageType::pixelType pixelType;
        typedef typename imageType::lineType lineType;
        typedef typename imageType::sliceType sliceType;

        tertiaryImageFunction() {}
        
        template <class T2>
        tertiaryImageFunction( const imageType &imIn, const T2 &value1, const T2 &value2, Image<T2> &ImOut )
        {
            this->retVal = this->_exec ( imIn, value1, value2, ImOut );
        }
        template <class T2>
        tertiaryImageFunction( const imageType &imIn, const T2 &value1, const Image<T2> &imIn2, Image<T2> &ImOut )
        {
            this->retVal = this->_exec ( imIn, value1, imIn2, ImOut );
        }
        template <class T2>
        tertiaryImageFunction( const imageType &imIn, const Image<T2> &imIn1, const T2 &value2, Image<T2> &ImOut )
        {
            this->retVal = this->_exec ( imIn, imIn1, value2, ImOut );
        }
        template <class T2>
        tertiaryImageFunction( const imageType &imIn, const Image<T2> &imIn1, const Image<T2> &imIn2, Image<T2> &ImOut )
        {
            this->retVal = this->_exec ( imIn, imIn1, imIn2, ImOut );
        }
        
        
        template <class T2>
        inline RES_T operator() ( const imageType &imIn1, const Image<T2> &imIn2, const Image<T2> &imIn3, Image<T2> &ImOut )
        {
            return this->_exec ( imIn1, imIn2, imIn3, ImOut );
        }
        template <class T2>
        inline RES_T operator() ( const imageType &imIn1, const T2 &value, const Image<T2> &imIn2, Image<T2> &ImOut )
        {
            return this->_exec ( imIn1, value, imIn2, ImOut );
        }
        template <class T2>
        inline RES_T operator() ( const imageType &imIn1, const Image<T2> &imIn2, const T2 &value, Image<T2> &ImOut )
        {
            return this->_exec ( imIn1, imIn2, value, ImOut );
        }
        template <class T2>
        inline RES_T operator() ( const imageType &imIn, const T2 &value1, const T2 &value2, Image<T2> &ImOut )
        {
            return this->_exec ( imIn, value1, value2, ImOut );
        }

        template <class T2> 
        RES_T _exec ( const imageType &imIn1, const Image<T2> &imIn2, const Image<T2> &imIn3, Image<T2> &imOut );
        template <class T2> 
        RES_T _exec ( const imageType &imIn1, const T2 &value, const Image<T2> &imIn2, Image<T2> &imOut );
        template <class T2> 
        RES_T _exec ( const imageType &imIn1, const Image<T2> &imIn2, const T2 &value, Image<T2> &imOut );
        template <class T2> 
        RES_T _exec ( const imageType &imIn, const T2 &value1, const T2 &value2, Image<T2> &imOut );
    //   protected:
        lineFunction_T lineFunction;
    };



} // namespace smil


#include "DBaseImageOperations.hxx"

#endif
