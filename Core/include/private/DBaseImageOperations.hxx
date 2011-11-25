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


#ifndef _BASE_IMAGE_OPERATIONS_HXX
#define _BASE_IMAGE_OPERATIONS_HXX

#include "DImage.hpp"
#include "DMemory.hpp"

template <class T>
struct fillLine;

template <class T>
inline typename Image<T>::lineType *imageFunctionBase<T>::createAlignedBuffers(UINT8 nbr, UINT32 len)
{
    if (alignedBuffers) 
    {
	if (nbr==bufferNumber && len==bufferLength)
	  return alignedBuffers;
	
	deleteAlignedBuffers();
    }
    
    
    bufferNumber = nbr;
    bufferLength = len;
    bufferSize = bufferLength * sizeof(T);
    
    alignedBuffers = new lineType[bufferNumber];
    for (int i=0;i<bufferNumber;i++)
      alignedBuffers[i] = createAlignedBuffer<T>(len);
    
    return alignedBuffers;
}
    

template <class T>
inline void imageFunctionBase<T>::deleteAlignedBuffers()
{
    if (!alignedBuffers) return;
    
    for (UINT i=0;i<bufferNumber;i++)
      deleteAlignedBuffer<T>(alignedBuffers[i]);
}

template <class T>
inline void imageFunctionBase<T>::copyLineToBuffer(T *line, UINT32 bufIndex)
{
    memcpy(alignedBuffers[bufIndex], line, bufferSize);
}

template <class T>
inline void imageFunctionBase<T>::copyBufferToLine(UINT32 bufIndex, T *line)
{
    memcpy(line, alignedBuffers[bufIndex], bufferSize);
}




template <class T, class lineFunction_T>
inline RES_T unaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn, imageType &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn.getLineCount();
    
    lineType *srcLines = imIn.getLines();
    lineType *destLines = imOut.getLines();
    
    for (int i=0;i<lineCount;i++)
	lineFunction(srcLines[i], lineLen, destLines[i]);
	
    imOut.modified();

    return RES_OK;
}


template <class T, class lineFunction_T>
inline RES_T unaryImageFunction<T, lineFunction_T>::_exec(imageType &imOut, T &value)
{
    if (!areAllocated(&imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imOut.getWidth();
    int lineCount = imOut.getLineCount();

    lineType *destLines = imOut.getLines();
    T *constBuf = createAlignedBuffer<T>(lineLen);
    
    // Fill the first aligned buffer with the constant value
    fillLine<T>::_exec(constBuf, lineLen, value);

    // Use it for operations on lines
    
    for (int i=0;i<lineCount;i++)
	lineFunction._exec_aligned(constBuf, lineLen, destLines[i]);
      
    deleteAlignedBuffer<T>(constBuf);
    imOut.modified();
}


// Binary image function
template <class T, class lineFunction_T>
inline RES_T binaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn1, imageType &imIn2, imageType &imOut)
{
    if (!areAllocated(&imIn1, &imIn2, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn1.getWidth();
    int lineCount = imIn1.getLineCount();
    
    lineType *srcLines1 = imIn1.getLines();
    lineType *srcLines2 = imIn2.getLines();
    lineType *destLines = imOut.getLines();
    
    for (int i=0;i<lineCount;i++)
	lineFunction(srcLines1[i], srcLines2[i], lineLen, destLines[i]);
      
    imOut.modified();

    return RES_OK;
}

// Binary image function
template <class T, class lineFunction_T>
inline RES_T binaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn, imageType &imInOut)
{
    if (!areAllocated(&imIn, &imInOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
    int lineCount = imIn.getLineCount();
    
    lineType *srcLines1 = imIn.getLines();
    lineType *srcLines2 = imInOut.getLines();
    
    T *tmpBuf = createAlignedBuffer<T>(lineLen);
    
    for (int i=0;i<lineCount;i++)
	lineFunction(srcLines1[i], srcLines2[i], lineLen, tmpBuf);
      
    deleteAlignedBuffer<T>(tmpBuf);
    imInOut.modified();

	return RES_OK;
}


// Binary image function
template <class T, class lineFunction_T>
inline RES_T binaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn, T value, imageType &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
    int lineCount = imIn.getLineCount();
    
    lineType *srcLines = imIn.getLines();
    lineType *destLines = imOut.getLines();
    
    T *constBuf = createAlignedBuffer<T>(lineLen);
    
    // Fill the const buffer with the value
    fillLine<T> f;
    f(constBuf, lineLen, value);

    for (int i=0;i<lineCount;i++)
	lineFunction(srcLines[i], constBuf, lineLen, destLines[i]);
      
    deleteAlignedBuffer<T>(constBuf);
    imOut.modified();

	return RES_OK;
}



// Tertiary image function
template <class T, class lineFunction_T>
inline RES_T tertiaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn1, imageType &imIn2, imageType &imIn3, imageType &imOut)
{
    if (!areAllocated(&imIn1, &imIn2, &imIn3, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn1.getWidth();
    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn1.getLineCount();
    
    lineType *srcLines1 = imIn1.getLines();
    lineType *srcLines2 = imIn2.getLines();
    lineType *srcLines3 = imIn3.getLines();
    lineType *destLines = imOut.getLines();
    
    for (int i=0;i<lineCount;i++)
	lineFunction(srcLines1[i], srcLines2[i], srcLines3[i], lineLen, destLines[i]);
    
    imOut.modified();

    return RES_OK;
}

// Tertiary image function
template <class T, class lineFunction_T>
inline RES_T tertiaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn1, imageType &imIn2, T value, imageType &imOut)
{
    if (!areAllocated(&imIn1, &imIn2, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn1.getWidth();
    int lineCount = imIn1.getLineCount();
    
    lineType *srcLines1 = imIn2.getLines();
    lineType *srcLines2 = imIn2.getLines();
    lineType *destLines = imOut.getLines();
    
    T *constBuf = createAlignedBuffer<T>(lineLen);
    
    // Fill the const buffer with the value
    fillLine<T> f;
    f(constBuf, lineLen, value);
    
    for (int i=0;i<lineCount;i++)
	lineFunction(srcLines1[i], srcLines2[i], constBuf, lineLen, destLines[i]);

    deleteAlignedBuffer<T>(constBuf);
    imOut.modified();

	return RES_OK;
}

template <class T, class lineFunction_T>
inline RES_T tertiaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn1, T value, imageType &imIn2, imageType &imOut)
{
    return tertiaryImageFunction<T, lineFunction_T>::_exec(imIn1, imIn2, value, imOut);
}


template <class T, class lineFunction_T>
inline RES_T tertiaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn, T value1, T value2, imageType &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
    int lineCount = imIn.getLineCount();
    
    lineType *srcLines = imIn.getLines();
    lineType *destLines = imOut.getLines();
    
    T *constBuf1 = createAlignedBuffer<T>(lineLen);
    T *constBuf2 = createAlignedBuffer<T>(lineLen);
    
    // Fill the const buffers with the values
    fillLine<T> f;
    f(constBuf1, lineLen, value1);
    f(constBuf2, lineLen, value2);
    
    for (int i=0;i<lineCount;i++)
	lineFunction(srcLines[i], constBuf1, constBuf2, lineLen, destLines[i]);
      
    deleteAlignedBuffer<T>(constBuf1);
    deleteAlignedBuffer<T>(constBuf2);
    imOut.modified();

    return RES_OK;
}



#endif
