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


#ifndef _D_IMAGE_ARITH_HPP
#define _D_IMAGE_ARITH_HPP

#include <typeinfo>

#include "DBaseImageOperations.hpp"
#include "DLineArith.hpp"
#include "Core/include/DTime.h"
#include "Core/include/private/DTraits.hpp"

namespace smil
{
  
/**
 * \ingroup Base
 * \defgroup Arith Arithmetic operations
 * @{
 */

    /**
    * Fill an image with a given value.
    *
    * \param imOut Output image.
    * \param value The value to fill.
    *
    * \vectorized
    * \parallelized
    * 
    * \see Image::operator<<
    * 
    */
    template <class T>
    RES_T fill(Image<T> &imOut, const T &value)
    {
        ASSERT_ALLOCATED(&imOut);

        return unaryImageFunction<T, fillLine<T> >(imOut, value).retVal;
    }

    /**
    * Fill an image with random values.
    *
    * \param imOut Output image.
    *
    * \see Image::operator<<
    * 
    */
    template <class T>
    RES_T randFill(Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imOut);

        typename ImDtTypes<T>::lineType pixels = imOut.getPixels();
        
        // Initialize random number generator
        struct timeval tv;
        gettimeofday(&tv, 0);
        srand(tv.tv_usec);
        
        double rangeT = ImDtTypes<T>::cardinal();
        T minT = ImDtTypes<T>::min();
        
        for (size_t i=0;i<imOut.getPixelCount();i++)
          pixels[i] = T( rand()/double(RAND_MAX) * rangeT + double(minT) );
        
        imOut.modified();
        
        return RES_OK;          
    }


    /**
    * Copy image
    * 
    * Copy an image (or a zone) into an output image
    * \param imIn input image
    * \param "startX startY [startZ]" (optional) start position of the zone in the input image
    * \param "sizeX sizeY [sizeZ]" (optional) size of the zone in the input image
    * \param imOut output image
    * \param "outStartX outStartY [outStartZ]" (optional) position to copy the selected zone in the output image (default is the origin (0,0,0))
    * 
    * \demo{copy_crop.py}
    */
    template <class T1, class T2>
    RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY, size_t startZ, size_t sizeX, size_t sizeY, size_t sizeZ, Image<T2> &imOut, size_t outStartX=0, size_t outStartY=0, size_t outStartZ=0)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);

        size_t inW = imIn.getWidth();
        size_t inH = imIn.getHeight();
        size_t inD = imIn.getDepth();
        
        size_t outW = imOut.getWidth();
        size_t outH = imOut.getHeight();
        size_t outD = imOut.getDepth();
        
        ASSERT(startX<inW && startY<inH && startZ<inD);
        ASSERT(outStartX<outW && outStartY<outH && outStartZ<outD);
        
        size_t realSx = min( min(sizeX, inW-startX), outW-outStartX );
        size_t realSy = min( min(sizeY, inH-startY), outH-outStartY );
        size_t realSz = min( min(sizeZ, inD-startZ), outD-outStartZ );

        typename Image<T1>::volType slIn = imIn.getSlices() + startZ;
        typename Image<T2>::volType slOut = imOut.getSlices() + outStartZ;
        
        size_t y;
        
        for (size_t z=0;z<realSz;z++)
        {
            typename Image<T1>::sliceType lnIn = *slIn + startY;
            typename Image<T2>::sliceType lnOut = *slOut + outStartY;
            
        
            #ifdef USE_OPEN_MP
                int nthreads = Core::getInstance()->getNumberOfThreads();
                #pragma omp parallel private(y)
            #endif // USE_OPEN_MP
            {
                #ifdef USE_OPEN_MP
                    #pragma omp for schedule(dynamic,nthreads) nowait
                #endif // USE_OPEN_MP
                for (y=0;y<realSy;y++)
                  copyLine<T1,T2>(lnIn[y]+startX, realSx, lnOut[y]+outStartX);
            }
            
            slIn++;
            slOut++;
        }
          
        imOut.modified();
        return RES_OK;
    }

    // 2D overload
    template <class T1, class T2>
    RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY, size_t sizeX, size_t sizeY, Image<T2> &imOut, size_t outStartX=0, size_t outStartY=0, size_t outStartZ=0)
    {
        return copy(imIn, startX, startY, 0, sizeX, sizeY, 1, imOut, outStartX, outStartY, outStartZ);
    }

    template <class T1, class T2>
    RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY, size_t startZ, Image<T2> &imOut, size_t outStartX=0, size_t outStartY=0, size_t outStartZ=0)
    {
        return copy(imIn, startX, startY, startZ, imIn.getWidth(), imIn.getHeight(), imIn.getDepth(), imOut, outStartX, outStartY, outStartZ);
    }

    // 2D overload
    template <class T1, class T2>
    RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY, Image<T2> &imOut, size_t outStartX=0, size_t outStartY=0, size_t outStartZ=0)
    {
        return copy(imIn, startX, startY, 0, imIn.getWidth(), imIn.getHeight(), 1, imOut, outStartX, outStartY, outStartZ);
    }


    template <class T1, class T2>
    RES_T copy(const Image<T1> &imIn, Image<T2> &imOut, size_t outStartX, size_t outStartY, size_t outStartZ=0)
    {
        return copy(imIn, 0, 0, 0, imIn.getWidth(), imIn.getHeight(), imIn.getDepth(), imOut, outStartX, outStartY, outStartZ);
    }


    // Copy/cast two images with different types but same size (quick way)
    template <class T1, class T2>
    RES_T copy(const Image<T1> &imIn, Image<T2> &imOut)
    {
        // Swig is (surprisingly;)) lost with overloads of template functions, so we try to reorient him
        if (typeid(imIn)==typeid(imOut))
          return copy(imIn, imOut);
        
        ASSERT_ALLOCATED(&imIn, &imOut);
      
        if (!haveSameSize(&imIn, &imOut, NULL))
            return copy<T1,T2>(imIn, 0, 0, 0, imOut, 0, 0, 0);

        copyLine<T1,T2>(imIn.getPixels(), imIn.getPixelCount(), imOut.getPixels());

        imOut.modified();
        return RES_OK;
    }

    template <class T>
    RES_T copy(const Image<T> &imIn, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        
        imOut.setSize(imIn);

        return unaryImageFunction<T, fillLine<T> >(imIn, imOut).retVal;
    }

    /**
    * Clone an image 
    * 
    * Set same size and copy contents
    */
    template <class T>
    RES_T clone(const Image<T> &imIn, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn);

        ASSERT((imOut.setSize(imIn)==RES_OK));
        return copy<T>(imIn, imOut);
    }

    /**
     * Cast from an image type to another
     * 
     */
    template <class T1, class T2>
    RES_T cast(const Image<T1> &imIn, Image<T2> &imOut)
    {
        ASSERT_ALLOCATED(&imIn);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        T1 floor_t1 = ImDtTypes<T1>::min();
        T2 floor_t2 = ImDtTypes<T2>::min();
        double coeff = double(ImDtTypes<T2>::cardinal()) / double(ImDtTypes<T1>::cardinal());
        
        typename Image<T1>::lineType pixIn = imIn.getPixels();
        typename Image<T2>::lineType pixOut = imOut.getPixels();
        
        size_t i, nPix = imIn.getPixelCount();
        
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<nPix;i++)
                pixOut[i] = floor_t2 + T2( coeff * double(pixIn[i]-floor_t1) );
        }
        
        return RES_OK;
    }

    /**
     * Copy a channel of multichannel image into a single channel image
     * \demo{multichannel_operations.py}
     */
    template <class MCT1, class T2>
    RES_T copyChannel(const Image<MCT1> &imIn, const UINT &chanNum, Image<T2> &imOut)
    {
        ASSERT(chanNum < MCT1::channelNumber());
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        typedef typename MCT1::DataType T1;
        typename Image<T1>::lineType lineIn = imIn.getPixels().arrays[chanNum];
        typename Image<T2>::lineType lineOut = imOut.getPixels();
        
        copyLine<T1,T2>(lineIn, imIn.getPixelCount(), lineOut);
        imOut.modified();
        return RES_OK;
    }
   
    /**
     * Copy a single channel image into a channel of multichannel image
     * \demo{multichannel_operations.py}
     */
    template <class T1, class MCT2>
    RES_T copyToChannel(const Image<T1> &imIn, const UINT &chanNum, Image<MCT2> &imOut)
    {
        ASSERT(chanNum < MCT2::channelNumber());
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        typedef typename MCT2::DataType T2;
        typename Image<T1>::lineType lineIn = imIn.getPixels();
        typename Image<T2>::lineType lineOut = imOut.getPixels().arrays[chanNum];
        
        copyLine<T1,T2>(lineIn, imIn.getPixelCount(), lineOut);
        imOut.modified();
        return RES_OK;
    }
   
    /**
     * Split channels of multichannel image to a 3D image with each channel on a Z slice
     * \demo{multichannel_operations.py}
     */
    template <class MCT1, class T2>
    RES_T splitChannels(const Image<MCT1> &imIn, Image<T2> &im3DOut)
    {
        ASSERT_ALLOCATED(&imIn);
        
        UINT width = imIn.getWidth(), height = imIn.getHeight();
        UINT chanNum = MCT1::channelNumber();
        UINT pixCount = width*height;
        ASSERT(im3DOut.setSize(width, height, chanNum)==RES_OK);

        typedef typename MCT1::DataType T1;
        typename Image<MCT1>::lineType lineIn = imIn.getPixels();
        typename Image<T2>::lineType lineOut = im3DOut.getPixels();
        
        for (UINT i=0;i<chanNum;i++)
        {
            copyLine<T1,T2>(lineIn.arrays[i], pixCount, lineOut);
            lineOut += pixCount;
        }
        im3DOut.modified();
        
        return RES_OK;
    }
   
    /**
     * Merge slices of a 3D image into a multichannel image
     * \demo{multichannel_operations.py}
     */
    template <class T1, class MCT2>
    RES_T mergeChannels(const Image<T1> &imIn, Image<MCT2> &imOut)
    {
        ASSERT_ALLOCATED(&imIn);
        UINT chanNum = MCT2::channelNumber();
        ASSERT(imIn.getDepth()==chanNum);
        
        UINT width = imIn.getWidth(), height = imIn.getHeight();
        UINT pixCount = width*height;
        imOut.setSize(width, height);

        typedef typename MCT2::DataType T2;
        typename Image<T1>::lineType lineIn = imIn.getPixels();
        typename Image<MCT2>::lineType lineOut = imOut.getPixels();
        
        for (UINT i=0;i<chanNum;i++)
        {
            copyLine<T1,T2>(lineIn, pixCount, lineOut.arrays[i]);
            lineIn += pixCount;
        }
        imOut.modified();
        
        return RES_OK;
    }
   
    /**
    * Invert an image.
    *
    * \param imIn Input image.
    * \param imOut Output image.
    *
    * \see Image::operator~
    */
    template <class T>
    RES_T inv(const Image<T> &imIn, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        return unaryImageFunction<T, invLine<T> >(imIn, imOut).retVal;
    }

    /**
    * Addition
    * 
    * Addition between two images (or between an image and a constant value)
    * \param imIn1 input image 1
    * \param imIn2 input image 2
    * \param imOut output image
    * \see Image::operator+
    */
    template <class T>
    RES_T add(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, addLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T add(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);
        
        return binaryImageFunction<T, addLine<T> >(imIn1, value, imOut);
    }

    /**
    * Addition (without saturation check)
    * 
    * Addition between two images (or between an image and a constant value) without checking the saturation
    * \param imIn1 input image 1
    * \param imIn2 input image 2
    * \param imOut output image
    */
    template <class T>
    RES_T addNoSat(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, addNoSatLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T addNoSat(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);
        
        return binaryImageFunction<T, addNoSatLine<T> >(imIn1, value, imOut);
    }


    /**
    * Subtraction
    * 
    * Subtraction between two images (or between an image and a constant value)
    * \param imIn1 input image 1
    * \param imIn2 input image 2
    * \param imOut output image containing \c imIn1-imIn2 (or \c imIn1-val)
    */
    template <class T>
    RES_T sub(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, subLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T sub(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        return binaryImageFunction<T, subLine<T> >(imIn1, value, imOut);
    }

    template <class T>
    RES_T sub(const T &value, const Image<T> &imIn, Image<T> &imOut)
    {
        return binaryImageFunction<T, subLine<T> >(value, imIn, imOut);
    }

    /**
    * Subtraction (without type minimum check)
    * 
    * Subtraction between two images (or between an image and a constant value)
    * \param imIn1 input image 1
    * \param imIn2 input image 2
    * \param imOut output image containing \c imIn1-imIn2 (or \c imIn1-val)
    */
    template <class T>
    RES_T subNoSat(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, subNoSatLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T subNoSat(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);
        
        return binaryImageFunction<T, subNoSatLine<T> >(imIn1, value, imOut);
    }

    template <class T>
    RES_T subNoSat(const T &value, const Image<T> &imIn, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        return binaryImageFunction<T, subNoSatLine<T> >(value, imIn, imOut);
    }

    /**
    * Sup of two images
    */
    template <class T>
    RES_T sup(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, supLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    ResImage<T> sup(const Image<T> &imIn1, const Image<T> &imIn2)
    {
        ResImage<T> newIm(imIn1);
        
        ASSERT(CHECK_ALLOCATED(&imIn1, &imIn2), newIm);
        ASSERT(CHECK_SAME_SIZE(&imIn1, &imIn2), newIm);
        
        sup(imIn1, imIn2, newIm);
        return newIm;
    }

    template <class T>
    RES_T sup(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);
        
        return binaryImageFunction<T, supLine<T> >(imIn1, value, imOut);
    }

    /**
    * Inf of two images
    */
    template <class T>
    RES_T inf(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);
        
        return binaryImageFunction<T, infLine<T> >(imIn1, value, imOut);
    }

    template <class T>
    ResImage<T> inf(const Image<T> &imIn1, const Image<T> &imIn2)
    {
        ResImage<T> newIm(imIn1);
        
        ASSERT(CHECK_ALLOCATED(&imIn1, &imIn2), newIm);
        ASSERT(CHECK_SAME_SIZE(&imIn1, &imIn2), newIm);
        
        inf(imIn1, imIn2, newIm);
        return newIm;
    }

    template <class T>
    RES_T inf(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, infLine<T> >(imIn1, imIn2, imOut);
    }

    /**
    * Equality operator
    * \return[imOut] image with imOut(x)=max(T) when imIn1(x)=imIn2(x) and 0 otherwise
    */
    template <class T>
    RES_T equ(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, equLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T equ(const Image<T> &imIn, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        return binaryImageFunction<T, equLine<T> >(imIn, value, imOut);
    }

    /**
    * Test equality between two images
    * \return True if imIn1=imIn2, False otherwise
    */
    template <class T>
    bool equ(const Image<T> &imIn1, const Image<T> &imIn2)
    {
        ASSERT(CHECK_ALLOCATED(&imIn1, &imIn2), false);
        ASSERT(CHECK_SAME_SIZE(&imIn1, &imIn2), false);
        
        typedef typename Image<T>::lineType lineType;
        lineType pix1 = imIn1.getPixels();
        lineType pix2 = imIn2.getPixels();
        
        for (size_t i=0;i<imIn1.getPixelCount();i++)
          if (pix1[i]!=pix2[i])
            return false;
          
        return true;
    }

    /**
    * Difference between two images.
    * 
    */
    template <class T>
    RES_T diff(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, diffLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T diff(const Image<T> &imIn, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        return binaryImageFunction<T, diffLine<T> >(imIn, value, imOut);
    }

    /**
    * Absolute difference ("vertical distance") between two images.
    * 
    * \return abs(p1-p2) for each pixels pair.
    */
    template <class T>
    RES_T absDiff(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);
        
        return binaryImageFunction<T, absDiffLine<T> >(imIn1, imIn2, imOut);
    }

    /**
    * Greater operator
    */
    template <class T>
    RES_T grt(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, grtLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T grt(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);

        return binaryImageFunction<T, grtLine<T> >(imIn1, value, imOut);
    }

    /**
    * Greater or equal operator
    */
    template <class T>
    RES_T grtOrEqu(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, grtOrEquLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T grtOrEqu(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);

        return binaryImageFunction<T, grtOrEquLine<T> >(imIn1, value, imOut);
    }

    /**
    * Lower operator
    */
    template <class T>
    RES_T low(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, lowLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T low(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);

        return binaryImageFunction<T, lowLine<T> >(imIn1, value, imOut);
    }

    /**
    * Lower or equal operator
    */
    template <class T>
    RES_T lowOrEqu(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, lowOrEquLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T lowOrEqu(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);

        return binaryImageFunction<T, lowOrEquLine<T> >(imIn1, value, imOut);
    }

    /**
    * Division
    */
    template <class T>
    RES_T div(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, divLine<T> >(imIn1, imIn2, imOut);
    }

//     template <class T>
//     RES_T div(const Image<T> &imIn, const T &value, Image<T> &imOut)
//     {
//         ASSERT_ALLOCATED(&imIn, &imOut);
//         ASSERT_SAME_SIZE(&imIn, &imOut);
// 
//         return binaryImageFunction<T, divLine<T> >(imIn, value, imOut);
//     }

    template <class T>
    RES_T div(const Image<T> &imIn, const double &dValue, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);

        typename ImDtTypes<T>::lineType pixIn = imIn.getPixels();
        typename ImDtTypes<T>::lineType pixOut = imOut.getPixels();
        
        for (size_t i=0;i<imIn.getPixelCount();i++)
          pixOut[i] = pixIn[i] / dValue;
        
        return RES_OK;
    }

    /**
    * Multiply
    */
    template <class T>
    RES_T mul(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, mulLine<T> >(imIn1, imIn2, imOut);
    }

//     template <class T>
//     RES_T mul(const Image<T> &imIn1, const T &value, Image<T> &imOut)
//     {
//         ASSERT_ALLOCATED(&imIn1, &imOut);
//         ASSERT_SAME_SIZE(&imIn1, &imOut);
// 
//         return binaryImageFunction<T, mulLine<T> >(imIn1, value, imOut);
//     }
    template <class T>
    RES_T mul(const Image<T> &imIn, const double &dValue, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);

        typename ImDtTypes<T>::lineType pixIn = imIn.getPixels();
        typename ImDtTypes<T>::lineType pixOut = imOut.getPixels();
        double newVal;
        
        for (size_t i=0;i<imIn.getPixelCount();i++)
        {
          newVal = pixIn[i] * dValue;
          pixOut[i] = newVal>double(ImDtTypes<T>::max()) ? ImDtTypes<T>::max() : T(newVal);
        }
        
        return RES_OK;
    }


    /**
    * Multiply (without type max check)
    */
    template <class T>
    RES_T mulNoSat(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, mulLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T mulNoSat(const Image<T> &imIn1, const T &value, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imOut);

        return binaryImageFunction<T, mulLine<T> >(imIn1, value, imOut);
    }

    /**
    * Logarithm
    * 
    * Possible bases: 0 or none (natural logarithm, or base e), 2, 10
    */
    template <class T>
    RES_T log(const Image<T> &imIn, Image<T> &imOut, int base=0)
    {
        ASSERT_ALLOCATED(&imIn);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        unaryImageFunction<T, logLine<T> > func;
        func.lineFunction.base = base;
        return func(imIn, imOut);
    }
    
    /**
    * Logic AND operator
    */
    template <class T>
    RES_T logicAnd(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, logicAndLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T bitAnd(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, bitAndLine<T> >(imIn1, imIn2, imOut);
    }

    /**
    * Logic OR operator
    */
    template <class T>
    RES_T logicOr(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, logicOrLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T bitOr(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, bitOrLine<T> >(imIn1, imIn2, imOut);
    }

    /**
    * Logic XOR operator
    */
    template <class T>
    RES_T logicXOr(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, logicXOrLine<T> >(imIn1, imIn2, imOut);
    }

    template <class T>
    RES_T bitXOr(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return binaryImageFunction<T, bitXOrLine<T> >(imIn1, imIn2, imOut);
    }

    /**
    * Test
    * 
    * If imIn1(x)!=0, imOut(x)=imIn2(x)\n
    * imOut(x)=imIn3(x) otherwise.
    * 
    * Can also be used with constant values and result of operators.
    * 
    * \par Examples
    * \code
    * test(im1>100, 255, 0, im2)
    * \endcode
    */
    template <class T1, class T2>
    RES_T test(const Image<T1> &imIn1, const Image<T2> &imIn2, const Image<T2> &imIn3, Image<T2> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imIn3, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imIn3, &imOut);

        return tertiaryImageFunction<T1, testLine<T1, T2> >(imIn1, imIn2, imIn3, imOut);
    }

    template <class T1, class T2>
    RES_T test(const Image<T1> &imIn1, const Image<T2> &imIn2, const T2 &value, Image<T2> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return tertiaryImageFunction<T1, testLine<T1, T2> >(imIn1, imIn2, value, imOut);
    }

    template <class T1, class T2>
    RES_T test(const Image<T1> &imIn1, const T2 &value, const Image<T2> &imIn2, Image<T2> &imOut)
    {
        ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
        ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

        return tertiaryImageFunction<T1, testLine<T1, T2> >(imIn1, value, imIn2, imOut);
    }

    template <class T1, class T2>
    RES_T test(const Image<T1> &imIn, const T2 &value1, const T2 &value2, Image<T2> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);

        return tertiaryImageFunction<T1, testLine<T1, T2> >(imIn, value1, value2, imOut);
    }

    template <class T1, class imOrValT, class trueT, class falseT, class T2>
    RES_T _compare_base(const Image<T1> &imIn, const char* compareType, const imOrValT &imOrVal, const trueT &trueImOrVal, const falseT &falseImOrVal, Image<T2> &imOut)
    {
        ImageFreezer freeze(imOut);
        
        Image<T1> tmpIm(imIn);
        
        if (strcmp(compareType, "==")==0)
        {
            ASSERT(equ(imIn, imOrVal, tmpIm)==RES_OK);
        }
        else if (strcmp(compareType, "!=")==0)
        {
            ASSERT(diff(imIn, imOrVal, tmpIm)==RES_OK);
        }
        else if (strcmp(compareType, ">")==0)
        {
            ASSERT(grt(imIn, imOrVal, tmpIm)==RES_OK);
        }
        else if (strcmp(compareType, "<")==0)
        {
            ASSERT(low(imIn, imOrVal, tmpIm)==RES_OK);
        }
        else if (strcmp(compareType, ">=")==0)
        {
            ASSERT(grtOrEqu(imIn, imOrVal, tmpIm)==RES_OK);
        }
        else if (strcmp(compareType, "<=")==0)
        {
            ASSERT(lowOrEqu(imIn, imOrVal, tmpIm)==RES_OK);
        }
        else 
        {
            ERR_MSG("Unknown operation");
            return RES_ERR;
        }

        ASSERT(test(tmpIm, trueImOrVal, falseImOrVal, imOut)==RES_OK);
        
        return RES_OK;
        
    }
    
    /**
     * Compare two images (or an image and a value)
     * 
     */
    template <class T1, class T2>
    RES_T compare(const Image<T1> &imIn1, const char* compareType, const Image<T1> &imIn2, const Image<T2> &trueIm, const Image<T2> &falseIm, Image<T2> &imOut)
    {
        return _compare_base<T1, Image<T1>, Image<T2>, Image<T2>, T2 >(imIn1, compareType, imIn2, trueIm, falseIm, imOut);
    }
    
    template <class T1, class T2>
    RES_T compare(const Image<T1> &imIn1, const char* compareType, const Image<T1> &imIn2, const T2 &trueVal, const Image<T2> &falseIm, Image<T2> &imOut)
    {
        return _compare_base<T1, Image<T1>, T2, Image<T2>, T2 >(imIn1, compareType, imIn2, trueVal, falseIm, imOut);
    }
    
    template <class T1, class T2>
    RES_T compare(const Image<T1> &imIn1, const char* compareType, const Image<T1> &imIn2, const Image<T2> &trueIm, const T2 &falseVal, Image<T2> &imOut)
    {
        return _compare_base<T1, Image<T1>, Image<T2>, T2, T2>(imIn1, compareType, imIn2, trueIm, falseVal, imOut);
    }
    
    template <class T1, class T2>
    RES_T compare(const Image<T1> &imIn1, const char* compareType, const Image<T1> &imIn2, const T2 &trueVal, const T2 &falseVal, Image<T2> &imOut)
    {
        return _compare_base<T1, Image<T1>, T2, T2, T2 >(imIn1, compareType, imIn2, trueVal, falseVal, imOut);
    }
    

    template <class T1, class T2>
    RES_T compare(const Image<T1> &imIn, const char* compareType, const T1 &value, const Image<T2> &trueIm, const Image<T2> &falseIm, Image<T2> &imOut)
    {
        return _compare_base<T1, T1, Image<T2>, Image<T2>, T2 >(imIn, compareType, value, trueIm, falseIm, imOut);
    }
    
    template <class T1, class T2>
    RES_T compare(const Image<T1> &imIn, const char* compareType, const T1 &value, const T2 &trueVal, const Image<T2> &falseIm, Image<T2> &imOut)
    {
        return _compare_base<T1, T1, T2, Image<T2>, T2 >(imIn, compareType, value, trueVal, falseIm, imOut);
    }
    
    template <class T1, class T2>
    RES_T compare(const Image<T1> &imIn, const char* compareType, const T1 &value, const Image<T2> &trueIm, const T2 &falseVal, Image<T2> &imOut)
    {
        return _compare_base<T1, T1, Image<T2>, T2, T2 >(imIn, compareType, value, trueIm, falseVal, imOut);
    }
    
    template <class T1, class T2>
    RES_T compare(const Image<T1> &imIn, const char* compareType, const T1 &value, const T2 &trueVal, const T2 &falseVal, Image<T2> &imOut)
    {
        return _compare_base<T1, T1, T2, T2, T2 >(imIn, compareType, value, trueVal, falseVal, imOut);
    }
    

    /**
     * Image mask
     * 
     * Returns an image imOut where 
     * - imOut(x)=imIn(x) if imMask(x)!=0
     * - imOut(x)=0 otherwise
     */
    template <class T>
    RES_T mask(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut)
    {
        return test<T>(imMask, imIn, T(0), imOut);
    }


    /**
    * Apply a lookup map
    * 
    * \b Python \b example:
    * \code{.py}
    * im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/balls.png")
    * imLbl = Image(im1, "UINT16")
    * imLbl2 = Image(imLbl)
    * 
    * label(im1, imLbl)
    * 
    * # We can use a Smil Map
    * # lookup = Map_UINT16_UINT16() 
    * # or directly a python dict
    * lookup = dict()
    * 
    * lookup[1] = 2
    * lookup[5] = 3
    * lookup[2] = 1
    * 
    * imLbl.showLabel()
    * imLbl2.showLabel()
    * 
    * \endcode
    */
    template <class T1, class mapT, class T2>
    RES_T applyLookup(const Image<T1> &imIn, const mapT &_map, Image<T2> &imOut, T2 defaultValue=T2(0))
    {
        ASSERT(!_map.empty(), "Input map is empty", RES_ERR);
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);

        // Verify that the max(measure) doesn't exceed the T2 type max
        typename mapT::const_iterator max_it = std::max_element(_map.begin(), _map.end(), map_comp_value_less());
        ASSERT(( max_it->second < ImDtTypes<T2>::max() ), "Input map max exceeds data type max!", RES_ERR);

        
        typename Image<T1>::lineType pixIn = imIn.getPixels();
        typename Image<T2>::lineType pixOut = imOut.getPixels();
        
        typename mapT::const_iterator it;
        
        for (size_t i=0;i<imIn.getPixelCount();i++)
        {
          it = _map.find(*pixIn);
          if (it!=_map.end())
            *pixOut = T2(it->second);
          else
            *pixOut = defaultValue;
          pixIn++;
          pixOut++;
        }
        imOut.modified();
        
        return RES_OK;
    }
    
    
#ifndef SWIG
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,UINT8) && !IS_SAME(T1,UINT16), RES_T ) // SFINAE General case
    applyLookup(const Image<T1> &imIn, const map<T1,T2> &lut, Image<T2> &imOut, T2 defaultValue=T2(0))
    {
        return applyLookup<T1, map<T1,T2>, T2>(imIn, lut, imOut, defaultValue);
    }

    // Specialization for T1==UINT8 or T1==UINT16
    template <class T1, class T2>
    ENABLE_IF( IS_SAME(T1,UINT8) || IS_SAME(T1,UINT16), RES_T ) // SFINAE For T1==UINT8 || T1==UINT16
    applyLookup(const Image<T1> &imIn, const map<T1,T2> &lut, Image<T2> &imOut, T2 defaultValue=T2(0))
    {
        ASSERT(!lut.empty(), "Input map is empty", RES_ERR);
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        T2 *outVals = ImDtTypes<T2>::createLine(ImDtTypes<T1>::cardinal());
        
        for (int i=0;i<ImDtTypes<T1>::max();i++)
          outVals[i] = defaultValue;
        
        typename Image<T1>::lineType pixIn = imIn.getPixels();
        typename Image<T2>::lineType pixOut = imOut.getPixels();
        
        for (typename map<T1,T2>::const_iterator it = lut.begin(); it!=lut.end(); it++)
          outVals[it->first] = it->second;
        
        for (size_t i=0;i<imIn.getPixelCount();i++)
          pixOut[i] = outVals[ pixIn[i] ];
        
        imOut.modified();
        
        ImDtTypes<T2>::deleteLine(outVals);

        return RES_OK;
    }
#else // SWIG
    template <class T1, class T2>
    RES_T applyLookup(const Image<T1> &imIn, const map<T1,T2> &lut, Image<T2> &imOut, T2 defaultValue=T2(0));
#endif // SWIG    

    
    
/** @}*/

} // namespace smil



#endif // _D_IMAGE_ARITH_HPP

