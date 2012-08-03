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


#ifndef _D_IMAGE_ARITH_HPP
#define _D_IMAGE_ARITH_HPP

#include "DBaseImageOperations.hpp"
#include "DLineArith.hpp"

/**
 * \ingroup Core
 * \defgroup Arith Base/arithmetic operations
 * @{
 */


template <class T>
RES_T fill(Image<T> &imOut, const T &value)
{
    if (!areAllocated(&imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    typedef typename Image<T>::sliceType sliceType;
    sliceType lineOut = imOut.getLines();
    int lineLen = imOut.getWidth();
    int lineCount = imOut.getLineCount();

    fillLine<T>(lineOut[0], lineLen, value);
    
    for (int i=1;i<lineCount;i++)
      copyLine<T>(lineOut[0], lineLen, lineOut[i]);

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
RES_T copy(const Image<T1> &imIn, UINT startX, UINT startY, UINT startZ, UINT sizeX, UINT sizeY, UINT sizeZ, Image<T2> &imOut, UINT outStartX=0, UINT outStartY=0, UINT outStartZ=0)
{
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    UINT inW = imIn.getWidth();
    UINT inH = imIn.getHeight();
    UINT inD = imIn.getDepth();
    
    UINT outW = imOut.getWidth();
    UINT outH = imOut.getHeight();
    UINT outD = imOut.getDepth();
    
    UINT realSx = min( min(sizeX, inW-startX), outW-outStartX );
    UINT realSy = min( min(sizeY, inH-startY), outH-outStartY );
    UINT realSz = min( min(sizeZ, inD-startZ), outD-outStartZ );

    typename Image<T1>::volType slIn = imIn.getSlices() + startZ;
    typename Image<T2>::volType slOut = imOut.getSlices() + outStartZ;
    
    for (UINT z=0;z<realSz;z++)
    {
	typename Image<T1>::sliceType lnIn = *slIn + startY;
	typename Image<T2>::sliceType lnOut = *slOut + outStartY;
	
	for (UINT y=0;y<realSy;y++)
	  copyLine<T1,T2>(lnIn[y]+startX, realSx, lnOut[y]+outStartX);
	
	slIn++;
	slOut++;
    }
      
    imOut.modified();
    return RES_OK;
}

// 2D overload
template <class T1, class T2>
RES_T copy(const Image<T1> &imIn, UINT startX, UINT startY, UINT sizeX, UINT sizeY, Image<T2> &imOut, UINT outStartX=0, UINT outStartY=0, UINT outStartZ=0)
{
    return copy(imIn, startX, startY, 0, sizeX, sizeY, 1, imOut, outStartX, outStartY, outStartZ);
}

template <class T1, class T2>
RES_T copy(const Image<T1> &imIn, UINT startX, UINT startY, UINT startZ, Image<T2> &imOut, UINT outStartX=0, UINT outStartY=0, UINT outStartZ=0)
{
    return copy(imIn, startX, startY, startZ, imIn.getWidth(), imIn.getHeight(), imIn.getDepth(), imOut, outStartX, outStartY, outStartZ);
}

// 2D overload
template <class T1, class T2>
RES_T copy(const Image<T1> &imIn, UINT startX, UINT startY, Image<T2> &imOut, UINT outStartX=0, UINT outStartY=0, UINT outStartZ=0)
{
    return copy(imIn, startX, startY, 0, imIn.getWidth(), imIn.getHeight(), 1, imOut, outStartX, outStartY, outStartZ);
}


template <class T1, class T2>
RES_T copy(const Image<T1> &imIn, Image<T2> &imOut, UINT outStartX, UINT outStartY, UINT outStartZ=0)
{
    return copy(imIn, 0, 0, 0, imIn.getWidth(), imIn.getHeight(), imIn.getDepth(), imOut, outStartX, outStartY, outStartZ);
}


// Copy/cast two images with different types but same size (quick way)
template <class T1, class T2>
RES_T copy(const Image<T1> &imIn, Image<T2> &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    if (!haveSameSize(&imIn, &imOut, NULL))
	return copy<T1,T2>(imIn, 0, 0, 0, imOut, 0, 0, 0);

    typename Image<T1>::sliceType l1 = imIn.getLines();
    typename Image<T2>::sliceType l2 = imOut.getLines();

    UINT width = imIn.getWidth();
    
    for (int i=0;i<imIn.getLineCount();i++)
      copyLine<T1,T2>(l1[i], width, l2[i]);

    imOut.modified();
    return RES_OK;
}

template <class T>
RES_T copy(const Image<T> &imIn, Image<T> &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    if (!haveSameSize(&imIn, &imOut, NULL))
	return copy<T,T>(imIn, 0, 0, 0, imOut, 0, 0, 0);

    typename Image<T>::lineType pixIn = imIn.getPixels();
    typename Image<T>::lineType pixOut = imOut.getPixels();
    
//     copyLine<T>(pixIn, imIn.getPixelCount(), pixOut);
    memcpy(pixOut, pixIn, imIn.getPixelCount());
}

/**
 * Crop image
 * 
 * Crop an image into an output image
 * \param imIn input image
 * \param "startX startY [startZ]" start position of the zone in the input image
 * \param "sizeX sizeY [sizeZ]" size of the zone in the input image
 * \param imOut output image
 * 
 * \demo{copy_crop.py}
 */
template <class T>
RES_T crop(const Image<T> &imIn, UINT startX, UINT startY, UINT startZ, UINT sizeX, UINT sizeY, UINT sizeZ, Image<T> &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    UINT inW = imIn.getWidth();
    UINT inH = imIn.getHeight();
    UINT inD = imIn.getDepth();
    
    UINT outW = imOut.getWidth();
    UINT outH = imOut.getHeight();
    UINT outD = imOut.getDepth();
    
    UINT realSx = min(sizeX, inW-startX);
    UINT realSy = min(sizeY, inH-startY);
    UINT realSz = min(sizeZ, inD-startZ);
    
    imOut.setSize(realSx, realSy, realSz);
    return copy(imIn, startX, startY, startZ, imOut, 0, 0, 0);
}

// 2D overload
template <class T>
RES_T crop(const Image<T> &imIn, UINT startX, UINT startY, UINT sizeX, UINT sizeY, Image<T> &imOut)
{
    return crop(imIn, startX, startY, 0, sizeX, sizeY, 1, imOut);
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
RES_T inv(Image<T> &imIn, Image<T> &imOut)
{
    return unaryImageFunction<T, invLine<T> >(imIn, imOut);
}

/**
 * Addition
 * 
 * Addition between two images (or between an image and a constant value)
 * \param imIn1 input image 1
 * \param "imIn2 (or val)" input image 2 (or constant value)
 * \param imOut output image
 * \see Image::operator+
 */
template <class T>
RES_T add(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, addLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T add(Image<T> &imIn1, const T &value, Image<T> &imOut)
{
    return binaryImageFunction<T, addLine<T> >(imIn1, value, imOut);
}

/**
 * Addition (without saturation check)
 * 
 * Addition between two images (or between an image and a constant value) without checking the saturation
 * \param imIn1 input image 1
 * \param "imIn2 (or val)" input image 2 (or constant value)
 * \param imOut output image
 */
template <class T>
RES_T addNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, addNoSatLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T addNoSat(Image<T> &imIn1, const T &value, Image<T> &imOut)
{
    return binaryImageFunction<T, addNoSatLine<T> >(imIn1, value, imOut);
}


/**
 * Subtraction
 * 
 * Subtraction between two images (or between an image and a constant value)
 * \param imIn1 input image 1
 * \param "imIn2 (or val)" input image 2 (or a constant value)
 * \param imOut output image containing \c imIn1-imIn2 (or \c imIn1-val)
 */
template <class T>
RES_T sub(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, subLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T sub(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, subLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T subNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, subNoSatLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T subNoSat(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, subNoSatLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T sup(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, supLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
Image<T>& sup(Image<T> &imIn1, Image<T> &imIn2)
{
    static Image<T> newIm(imIn1);
    sup(imIn1, imIn2, newIm);
    return newIm;
}

template <class T>
RES_T sup(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, supLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T inf(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, infLine<T> >(imIn1, value, imOut);
}

template <class T>
Image<T>& inf(Image<T> &imIn1, Image<T> &imIn2)
{
    static Image<T> newIm(imIn1);
    inf(imIn1, imIn2, newIm);
    return newIm;
}

template <class T>
RES_T inf(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, infLine<T> >(imIn1, imIn2, imOut);
}

/**
 * Test equality between two images
 * \return[imOut] image with max(T) when true and 0 otherwise
 */
template <class T>
RES_T equ(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, equLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
bool equ(Image<T> &imIn1, Image<T> &imIn2)
{
    typedef typename Image<T>::lineType lineType;
    lineType pix1 = imIn1.getPixels();
    lineType pix2 = imIn2.getPixels();
    
    for (int i=0;i<imIn1.getPixelCount();i++)
      if (pix1[i]!=pix2[i])
	return false;
      
    return true;
}

/**
 * Difference ("vertical distance") between two images.
 * 
 * \return abs(p1-p2) for each pixels pair.
 */
template <class T>
RES_T diff(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, diffLine<T> >(imIn1, imIn2, imOut);
}


template <class T>
RES_T grt(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, grtLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T grt(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, grtLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T grtOrEqu(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, grtOrEquLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T grtOrEqu(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, grtOrEquLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T low(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, lowLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T low(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, lowLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T lowOrEqu(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, lowOrEquLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T lowOrEqu(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, lowLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T div(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, divLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T div(Image<T> &imIn, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, divLine<T> >(imIn, value, imOut);
}

template <class T>
RES_T mul(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, mulLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T mul(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, mulLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T mulNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, mulLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T mulNoSat(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, mulLine<T> >(imIn1, value, imOut);
}

template <class T>
RES_T logicAnd(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, logicAndLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T logicOr(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, logicOrLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T logicXOr(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, logicXOrLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
RES_T test(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imIn3, Image<T> &imOut)
{
    return tertiaryImageFunction<T, testLine<T> >(imIn1, imIn2, imIn3, imOut);
}

template <class T>
RES_T test(Image<T> &imIn1, Image<T> &imIn2, T value, Image<T> &imOut)
{
    return tertiaryImageFunction<T, testLine<T> >(imIn1, imIn2, value, imOut);
}

template <class T>
RES_T test(Image<T> &imIn1, T value, Image<T> &imIn2, Image<T> &imOut)
{
    return tertiaryImageFunction<T, testLine<T> >(imIn1, value, imIn2, imOut);
}

template <class T>
RES_T test(Image<T> &imIn, T value1, T value2, Image<T> &imOut)
{
    return tertiaryImageFunction<T, testLine<T> >(imIn, value1, value2, imOut);
}



/**
 * Volume of an image
 *
 * Returns the sum of the pixel values.
 * \param imIn Input image.
 */
template <class T>
double vol(Image<T> &imIn)
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    int npix = imIn.getPixelCount();
    typename ImDtTypes<T>::lineType pixels = imIn.getPixels();
    double vol = 0;

    for (int i=0;i<npix;i++)
        vol += pixels[i];

    return vol;
}

/**
 * Min value of an image
 *
 * Returns the min of the pixel values.
 * \param imIn Input image.
 */
template <class T>
T minVal(Image<T> &imIn)
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    int npix = imIn.getPixelCount();
    typename ImDtTypes<T>::lineType p = imIn.getPixels();
    T minVal = numeric_limits<T>::max();

    for (int i=0;i<npix;i++,p++)
        if (*p<minVal)
            minVal = *p;

    return minVal;
}

/**
 * Max value of an image
 *
 * Returns the min of the pixel values.
 * \param imIn Input image.
 */
template <class T>
T maxVal(Image<T> &imIn)
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    int npix = imIn.getPixelCount();
    typename ImDtTypes<T>::lineType p = imIn.getPixels();
    T maxVal = numeric_limits<T>::min();

    for (int i=0;i<npix;i++,p++)
        if (*p>maxVal)
            maxVal = *p;

    return maxVal;
}

/**
 * Min and Max values of an image
 *
 * Returns the min and the max of the pixel values.
 * \param imIn Input image.
 */
template <class T>
void rangeVal(Image<T> &imIn, T *ret_min, T *ret_max)
{
    if (!imIn.isAllocated())
    {
	*ret_min = 0;
	*ret_max = 0;
        return;
    }

    int npix = imIn.getPixelCount();
    typename ImDtTypes<T>::lineType p = imIn.getPixels();
    *ret_min = numeric_limits<T>::max();
    *ret_max = numeric_limits<T>::min();

    for (int i=0;i<npix;i++,p++)
    {
        if (*p<*ret_min)
            *ret_min = *p;
        if (*p>*ret_max)
            *ret_max = *p;
    }

}

/** @}*/

#endif // _D_IMAGE_ARITH_HPP

