/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


#ifndef _BASE_IMAGE_OPERATIONS_HPP
#define _BASE_IMAGE_OPERATIONS_HPP

#include "DImage.hpp"
#include "DMemory.hpp"

#include "DBaseLineOperations.hpp"


template <class T>
class imageFunctionBase
{
  public:
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::pixelType pixelType;
    
    lineType *alignedBuffers;
    UINT32 bufferNumber;
    UINT32 bufferLength;
    UINT32 bufferSize;
    
    imageFunctionBase() : alignedBuffers(NULL) {};
    ~imageFunctionBase() 
    {
	deleteAlignedBuffers();
    };
    
    inline lineType *createAlignedBuffers(UINT8 nbr, UINT32 len);
    inline void deleteAlignedBuffers();
    inline void copyLineToBuffer(T *line, UINT32 bufIndex); 
    inline void copyBufferToLine(UINT32 bufIndex, T *line);  
};

template <class T, class lineFunction_T>
class unaryImageFunction : public imageFunctionBase<T>
{
  public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::pixelType pixelType;
    
    unaryImageFunction() {}
    inline RES_T operator()(imageType &imIn, imageType &ImOut)
    {
	return this->_exec(imIn, ImOut);
    }
    inline RES_T operator()(imageType &ImOut, T &value)
    {
	return this->_exec(ImOut, value);
    }
    
    inline RES_T _exec(imageType &imIn, imageType &imOut);
    inline RES_T _exec(imageType &imOut, T &value);
    
//   protected:	    
    lineFunction_T lineFunction;
};


template <class T, class lineFunction_T>
class binaryImageFunction : public imageFunctionBase<T>
{
  public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::pixelType pixelType;
    
    binaryImageFunction() {}
    binaryImageFunction(imageType &imIn1, imageType &imIn2, imageType &ImOut) { this->_exec(imIn1, imIn2, ImOut); }
    binaryImageFunction(imageType &imIn, T value, imageType &ImOut) { this->_exec(imIn, value, ImOut); }
    inline RES_T operator()(imageType &imIn1, imageType &imIn2, imageType &ImOut) { return this->_exec(imIn1, imIn2, ImOut); }
    inline RES_T operator()(imageType &imIn, T value, imageType &ImOut) { return this->_exec(imIn, value, ImOut); }
    
    inline RES_T _exec(imageType &imIn1, imageType &imIn2, imageType &imOut);
    inline RES_T _exec(imageType &imIn, imageType &imInOut);
    inline RES_T _exec(imageType &imIn, T value, imageType &imOut);
    
//   protected:	    
    lineFunction_T lineFunction;
};


template <class T, class lineFunction_T>
class tertiaryImageFunction : public imageFunctionBase<T>
{
  public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::pixelType pixelType;
    
    tertiaryImageFunction() {}
    inline RES_T operator()(imageType &imIn1, imageType &imIn2, imageType &imIn3, imageType &ImOut) { return this->_exec(imIn1, imIn2, imIn3, ImOut); }
    inline RES_T operator()(imageType &imIn1, T value, imageType &imIn2, imageType &ImOut) { return this->_exec(imIn1, value, imIn2, ImOut); }
    inline RES_T operator()(imageType &imIn1, imageType &imIn2, T value, imageType &ImOut) { return this->_exec(imIn1, imIn2, value, ImOut); }
    inline RES_T operator()(imageType &imIn, T value1, T value2, imageType &ImOut) { return this->_exec(imIn, value1, value2, ImOut); }
    
    inline RES_T _exec(imageType &imIn1, imageType &imIn2, imageType &imIn3, imageType &imOut);
//     static RES_T _exec(imageType &imIn1, imageType &imInOut);
    inline RES_T _exec(imageType &imIn1, T value, imageType &imIn2, imageType &imOut);
    inline RES_T _exec(imageType &imIn1, imageType &imIn2, T value, imageType &imOut);
    inline RES_T _exec(imageType &imIn, T value1, T value2, imageType &imOut);
    
//   protected:	    
    lineFunction_T lineFunction;
};






#include "DBaseImageOperations.hxx"

#endif
