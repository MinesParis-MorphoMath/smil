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


#ifndef _BASE_IMAGE_OPERATIONS_HXX
#define _BASE_IMAGE_OPERATIONS_HXX

#include "DImage.hpp"
#include "DMemory.hpp"

#include <omp.h>

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

#pragma omp parallel for
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

#pragma omp parallel for
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

    int i, chunk = 100;
    
    #pragma omp parallel shared(srcLines1,srcLines2,destLines,chunk) private(i)
    {
	#pragma omp for schedule(dynamic,chunk) nowait
	for (i=0;i<lineCount;i++)
	    lineFunction(srcLines1[i], srcLines2[i], lineLen, destLines[i]);
    }
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

#pragma omp parallel for
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

#pragma omp parallel for
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

#pragma omp parallel for
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

#pragma omp parallel for
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

#pragma omp parallel for
    for (int i=0;i<lineCount;i++)
        lineFunction(srcLines[i], constBuf1, constBuf2, lineLen, destLines[i]);

    deleteAlignedBuffer<T>(constBuf1);
    deleteAlignedBuffer<T>(constBuf2);
    imOut.modified();

    return RES_OK;
}



#endif
