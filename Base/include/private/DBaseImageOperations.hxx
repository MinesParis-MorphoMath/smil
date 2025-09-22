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


#ifndef _BASE_IMAGE_OPERATIONS_HXX
#define _BASE_IMAGE_OPERATIONS_HXX

#include "Core/include/private/DImage.hpp"

namespace smil
{
  
    template <class T>
    struct fillLine;

    template <class T>
    typename Image<T>::lineType *imageFunctionBase<T>::createAlignedBuffers(size_t nbr, size_t len)
    {
        if (alignedBuffers)
        {
            if (nbr==bufferNumber && len==bufferLength)
                return alignedBuffers;
            else
                deleteAlignedBuffers();
        }


        bufferNumber = nbr;
        bufferLength = len;
        bufferSize = bufferLength * sizeof(T);

        alignedBuffers = new lineType[bufferNumber];
        for (size_t i=0;i<bufferNumber;i++)
            alignedBuffers[i] = ImDtTypes<T>::createLine(len);

        return alignedBuffers;
    }


    template <class T>
    void imageFunctionBase<T>::deleteAlignedBuffers()
    {
        if (!alignedBuffers) return;

        for (UINT i=0;i<bufferNumber;i++)
          ImDtTypes<T>::deleteLine(alignedBuffers[i]);
        delete[] alignedBuffers;
        alignedBuffers = NULL;
    }

    template <class T>
    inline void imageFunctionBase<T>::copyLineToBuffer(T *line, size_t bufIndex)
    {
        memcpy(alignedBuffers[bufIndex], line, bufferSize);
    }

    template <class T>
    inline void imageFunctionBase<T>::copyBufferToLine(size_t bufIndex, T *line)
    {
        memcpy(line, alignedBuffers[bufIndex], bufferSize);
    }




    template <class T, class lineFunction_T, class T_out>
    RES_T unaryImageFunction<T, lineFunction_T, T_out>::_exec(const imageInType &imIn, imageOutType &imOut)
    {
        if (!areAllocated(&imIn, &imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        int lineLen = imIn.getWidth();
        int lineCount = imIn.getLineCount();

        sliceInType srcLines = imIn.getLines();
        sliceOutType destLines = imOut.getLines();
        
        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<lineCount;i++)
                lineFunction._exec(srcLines[i], lineLen, destLines[i]);
        }
        imOut.modified();

        return RES_OK;
    }


    template <class T, class lineFunction_T, class T_out>
    RES_T unaryImageFunction<T, lineFunction_T, T_out>::_exec(imageOutType &imOut, const T_out &value)
    {
        if (!areAllocated(&imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imOut.getWidth();
        int lineCount = imOut.getLineCount();

        sliceOutType destLines = imOut.getLines();
        lineOutType constBuf = ImDtTypes<T_out>::createLine(lineLen);

        // Fill the first aligned buffer with the constant value
        fillLine<T_out>(constBuf, lineLen, value);

        // Use it for operations on lines

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<lineCount;i++)
                lineFunction._exec(constBuf, lineLen, destLines[i]);
        }
        ImDtTypes<T_out>::deleteLine(constBuf);
        imOut.modified();
        
        return RES_OK;
    }


    // Binary image function
    template <class T, class lineFunction_T>
    RES_T binaryImageFunction<T, lineFunction_T>::_exec(const imageType &imIn1, const imageType &imIn2, imageType &imOut)
    {
        if (!areAllocated(&imIn1, &imIn2, &imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imIn1.getWidth();
        size_t lineCount = imIn1.getLineCount();

        lineType *srcLines1 = imIn1.getLines();
        lineType *srcLines2 = imIn2.getLines();
        lineType *destLines = imOut.getLines();

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<(int)lineCount;i++)
                    lineFunction(srcLines1[i], srcLines2[i], lineLen, destLines[i]);
        }
        imOut.modified();

        return RES_OK;
    }

    // Binary image function
    template <class T, class lineFunction_T>
    RES_T binaryImageFunction<T, lineFunction_T>::_exec(const imageType &imIn, imageType &imInOut)
    {
        if (!areAllocated(&imIn, &imInOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imIn.getWidth();
        int lineCount = imIn.getLineCount();

        sliceType srcLines1 = imIn.getLines();
        sliceType srcLines2 = imInOut.getLines();

        lineType tmpBuf = ImDtTypes<T>::createLine(lineLen);

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<lineCount;i++)
                lineFunction(srcLines1[i], srcLines2[i], lineLen, tmpBuf);
        }

        ImDtTypes<T>::deleteLine(tmpBuf);
        imInOut.modified();

        return RES_OK;
    }


    // Binary image function
    template <class T, class lineFunction_T>
    RES_T binaryImageFunction<T, lineFunction_T>::_exec(const imageType &imIn, const T &value, imageType &imOut)
    {
        if (!areAllocated(&imIn, &imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imIn.getWidth();
        int lineCount = imIn.getLineCount();

        sliceType srcLines = imIn.getLines();
        sliceType destLines = imOut.getLines();

        lineType constBuf = ImDtTypes<T>::createLine(lineLen);

        // Fill the const buffer with the value
        fillLine<T> f;
        f(constBuf, lineLen, value);

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
          for (i=0;i<lineCount;i++)
              lineFunction(srcLines[i], constBuf, lineLen, destLines[i]);
        }
        
        ImDtTypes<T>::deleteLine(constBuf);
        imOut.modified();

        return RES_OK;
    }


    template <class T, class lineFunction_T>
    RES_T binaryImageFunction<T, lineFunction_T>::_exec(const T &value, const imageType &imIn, imageType &imOut)
    {
        if (!areAllocated(&imIn, &imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imIn.getWidth();
        int lineCount = imIn.getLineCount();

        sliceType srcLines = imIn.getLines();
        sliceType destLines = imOut.getLines();

        lineType constBuf = ImDtTypes<T>::createLine(lineLen);

        // Fill the const buffer with the value
        fillLine<T> f;
        f(constBuf, lineLen, value);

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
          for (i=0;i<lineCount;i++)
              lineFunction(constBuf, srcLines[i], lineLen, destLines[i]);
        }
        
        ImDtTypes<T>::deleteLine(constBuf);
        imOut.modified();

        return RES_OK;
    }



    // Tertiary image function
    template <class T, class lineFunction_T>
    template <class T2> 
    RES_T tertiaryImageFunction<T, lineFunction_T>::_exec(const imageType &imIn1, const Image<T2> &imIn2, const Image<T2> &imIn3, Image<T2> &imOut)
    {
        if (!areAllocated(&imIn1, &imIn2, &imIn3, &imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imIn1.getWidth();
        int lineCount = imIn1.getLineCount();

        typedef typename Image<T2>::sliceType sliceType2;

        sliceType srcLines1 = imIn1.getLines();
        sliceType2 srcLines2 = imIn2.getLines();
        sliceType2 srcLines3 = imIn3.getLines();
        sliceType2 destLines = imOut.getLines();

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<lineCount;i++)
                lineFunction(srcLines1[i], srcLines2[i], srcLines3[i], lineLen, destLines[i]);
        }
            
        imOut.modified();

        return RES_OK;
    }

    // Tertiary image function
    template <class T, class lineFunction_T>
    template <class T2> 
    RES_T tertiaryImageFunction<T, lineFunction_T>::_exec(const imageType &imIn1, const Image<T2> &imIn2, const T2 &value, Image<T2> &imOut)
    {
        if (!areAllocated(&imIn1, &imIn2, &imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imIn1.getWidth();
        int lineCount = imIn1.getLineCount();

        typedef typename Image<T2>::lineType lineType2;
        typedef typename Image<T2>::sliceType sliceType2;
        
        sliceType srcLines1 = imIn1.getLines();
        sliceType2 srcLines2 = imIn2.getLines();
        sliceType2 destLines = imOut.getLines();

        lineType2 constBuf = ImDtTypes<T2>::createLine(lineLen);

        // Fill the const buffer with the value
        fillLine<T2> f;
        f(constBuf, lineLen, value);

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<lineCount;i++)
                lineFunction(srcLines1[i], srcLines2[i], constBuf, lineLen, destLines[i]);
        }
            
        ImDtTypes<T2>::deleteLine(constBuf);
        imOut.modified();

        return RES_OK;
    }

    template <class T, class lineFunction_T>
    template <class T2> 
    RES_T tertiaryImageFunction<T, lineFunction_T>::_exec(const imageType &imIn1, const T2 &value, const Image<T2> &imIn2, Image<T2> &imOut)
    {
        if (!areAllocated(&imIn1, &imIn2, &imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imIn1.getWidth();
        int lineCount = imIn1.getLineCount();

        typedef typename Image<T2>::lineType lineType2;
        typedef typename Image<T2>::sliceType sliceType2;
        
        sliceType srcLines1 = imIn1.getLines();
        sliceType2 srcLines2 = imIn2.getLines();
        sliceType2 destLines = imOut.getLines();

        lineType2 constBuf = ImDtTypes<T2>::createLine(lineLen);

        // Fill the const buffer with the value
        fillLine<T2> f;
        f(constBuf, lineLen, value);

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<lineCount;i++)
                lineFunction(srcLines1[i], constBuf, srcLines2[i], lineLen, destLines[i]);
        }

        ImDtTypes<T2>::deleteLine(constBuf);
        imOut.modified();

        return RES_OK;
    }


    template <class T, class lineFunction_T>
    template <class T2> 
    RES_T tertiaryImageFunction<T, lineFunction_T>::_exec(const imageType &imIn, const T2 &value1, const T2 &value2, Image<T2> &imOut)
    {
        if (!areAllocated(&imIn, &imOut, NULL))
            return RES_ERR_BAD_ALLOCATION;

        size_t lineLen = imIn.getWidth();
        int lineCount = imIn.getLineCount();

        typedef typename Image<T2>::lineType lineType2;
        typedef typename Image<T2>::sliceType sliceType2;
        
        sliceType srcLines = imIn.getLines();
        sliceType2 destLines = imOut.getLines();

        lineType2 constBuf1 = ImDtTypes<T2>::createLine(lineLen);
        lineType2 constBuf2 = ImDtTypes<T2>::createLine(lineLen);

        // Fill the const buffers with the values
        fillLine<T2> f;
        f(constBuf1, lineLen, value1);
        f(constBuf2, lineLen, value2);

        int i;
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<lineCount;i++)
                lineFunction(srcLines[i], constBuf1, constBuf2, lineLen, destLines[i]);
        }

        ImDtTypes<T2>::deleteLine(constBuf1);
        ImDtTypes<T2>::deleteLine(constBuf2);
        imOut.modified();

        return RES_OK;
    }

} // namespace smil


#endif
