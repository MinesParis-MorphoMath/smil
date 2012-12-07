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


#ifndef _BASE_IMAGE_OPERATIONS_HPP
#define _BASE_IMAGE_OPERATIONS_HPP

#include "Core/include/private/DImage.hpp"
#include "Core/include/private/DMemory.hpp"

#include "DBaseLineOperations.hpp"


template <class T>
class imageFunctionBase
{
public:
    typedef Image<T> imageType;
    typedef typename imageType::sliceType sliceType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::pixelType pixelType;

    sliceType alignedBuffers;
    UINT32 bufferNumber;
    UINT32 bufferLength;
    UINT32 bufferSize;
    
    RES_T retVal;

    imageFunctionBase() : alignedBuffers ( NULL ) {};
    ~imageFunctionBase()
    {
        deleteAlignedBuffers();
    };

    inline lineType *createAlignedBuffers ( UINT8 nbr, UINT32 len );
    inline void deleteAlignedBuffers();
    inline void copyLineToBuffer ( T *line, UINT32 bufIndex );
    inline void copyBufferToLine ( UINT32 bufIndex, T *line );
    
    operator RES_T() { return retVal; }
};

template <class T, class lineFunction_T>
class unaryImageFunction : public imageFunctionBase<T>
{
public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::pixelType pixelType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::sliceType sliceType;

    unaryImageFunction() {}
    unaryImageFunction( const imageType &imIn, imageType &ImOut ) 
    {
        this->retVal = this->_exec ( imIn, ImOut );
    }
    unaryImageFunction( const imageType &imIn, const T &value ) 
    {
        this->retVal = this->_exec (imIn, value);
    }
    
    inline RES_T operator() ( const imageType &imIn, imageType &ImOut )
    {
        return this->_exec ( imIn, ImOut );
    }
    inline RES_T operator() ( imageType &ImOut, T &value )
    {
        return this->_exec ( ImOut, value );
    }

    inline RES_T _exec ( const imageType &imIn, imageType &imOut );
    inline RES_T _exec ( imageType &imOut, const T &value );

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
    
    inline RES_T operator() ( const imageType &imIn1, const imageType &imIn2, imageType &ImOut )
    {
        return this->_exec ( imIn1, imIn2, ImOut );
    }
    inline RES_T operator() ( const imageType &imIn, const T &value, imageType &ImOut )
    {
        return this->_exec ( imIn, value, ImOut );
    }

    inline RES_T _exec ( const imageType &imIn1, const imageType &imIn2, imageType &imOut );
    inline RES_T _exec ( const imageType &imIn, imageType &imInOut );
    inline RES_T _exec ( const imageType &imIn, const T &value, imageType &imOut );

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
    tertiaryImageFunction( const imageType &imIn1, const imageType &imIn2, const imageType &imIn3, imageType &ImOut )
    {
        this->retVal = this->_exec ( imIn1, imIn2, imIn3, ImOut );
    }
    tertiaryImageFunction( const imageType &imIn1, const T &value, const imageType &imIn2, imageType &ImOut )
    {
        this->retVal = this->_exec ( imIn1, value, imIn2, ImOut );
    }
    tertiaryImageFunction( const imageType &imIn1, const imageType &imIn2, const T &value, imageType &ImOut )
    {
        this->retVal = this->_exec ( imIn1, imIn2, value, ImOut );
    }
    tertiaryImageFunction( const imageType &imIn, const T &value1, const T &value2, imageType &ImOut )
    {
        this->retVal = this->_exec ( imIn, value1, value2, ImOut );
    }
    
    
    inline RES_T operator() ( const imageType &imIn1, const imageType &imIn2, const imageType &imIn3, imageType &ImOut )
    {
        return this->_exec ( imIn1, imIn2, imIn3, ImOut );
    }
    inline RES_T operator() ( const imageType &imIn1, const T &value, const imageType &imIn2, imageType &ImOut )
    {
        return this->_exec ( imIn1, value, imIn2, ImOut );
    }
    inline RES_T operator() ( const imageType &imIn1, const imageType &imIn2, const T &value, imageType &ImOut )
    {
        return this->_exec ( imIn1, imIn2, value, ImOut );
    }
    inline RES_T operator() ( const imageType &imIn, const T &value1, const T &value2, imageType &ImOut )
    {
        return this->_exec ( imIn, value1, value2, ImOut );
    }

    inline RES_T _exec ( const imageType &imIn1, const imageType &imIn2, const imageType &imIn3, imageType &imOut );
//     static RES_T _exec(imageType &imIn1, imageType &imInOut);
    inline RES_T _exec ( const imageType &imIn1, const T &value, const imageType &imIn2, imageType &imOut );
    inline RES_T _exec ( const imageType &imIn1, const imageType &imIn2, const T &value, imageType &imOut );
    inline RES_T _exec ( const imageType &imIn, const T &value1, const T &value2, imageType &imOut );

//   protected:
    lineFunction_T lineFunction;
};






#include "DBaseImageOperations.hxx"

#endif
