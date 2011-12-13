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


#ifndef _D_IMAGE_ARITH_HPP
#define _D_IMAGE_ARITH_HPP

#include "DLineArith.hpp"

/**
 * \ingroup Core
 * \defgroup Arith
 * @{
 */


template <class T>
inline RES_T fill(Image<T> &imOut, const T value)
{
    if (!areAllocated(&imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    typedef typename Image<T>::lineType lineType;
    lineType *lineOut = imOut.getLines();
    int lineLen = imOut.getAllocatedWidth();
    int lineCount = imOut.getLineCount();

    fillLine<T> f;
    f(imOut.getPixels(), lineLen*lineCount, value);

    imOut.modified();
    return RES_OK;
}

//! Copy/cast (two images with different types)
template <class T1, class T2>
RES_T copy(Image<T1> &imIn, Image<T2> &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    if (haveSameSize(&imIn, &imOut, NULL))
    {
        typename Image<T1>::lineType *l1 = imIn.getLines();
        typename Image<T2>::lineType *l2 = imOut.getLines();

	UINT width = imIn.getAllocatedWidth();
	
        for (int i=0;i<imIn.getLineCount();i++)
	  copyLine<T1,T2>(l1[i], width, l2[i]);

        imOut.modified();
        return RES_OK;
    }
}

//! Copy (two images of same type)
template <class T>
RES_T copy(Image<T> &imIn, Image<T> &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    if (haveSameSize(&imIn, &imOut, NULL))
    {
// 	for (int j=0;j<imIn.getLineCount();j++)
// 	  copyLine(imIn.getLines()[j], imIn.getAllocatedWidth(), imOut.getLines()[j]);
        memcpy(imOut.getPixels(), imIn.getPixels(), imIn.getAllocatedSize());

        imOut.modified();
        return RES_OK;
    }
    return RES_OK;
}


/**
 * Invert an image.
 *
 * \param imIn Input image.
 * \param imOut Output image.
 *
 * \sa Image::operator<<
 */
template <class T>
inline RES_T inv(Image<T> &imIn, Image<T> &imOut)
{
    return unaryImageFunction<T, invLine<T> >(imIn, imOut);
}

/**
 * Add two images.
 *
 * \param "imIn1 imIn2" Input images.
 * \param imOut Output image.
 * \see addNoSat
 */
template <class T>
inline RES_T add(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, addLine<T> >(imIn1, imIn2, imOut);
}

/**
 * Add a constant value to an image.
 *
 * \param imIn Input image.
 * \param value The constant value to add.
 * \param imOut Output image.
 * \see addNoSat
 */
template <class T>
inline RES_T add(Image<T> &imIn1, const T value, Image<T> &imOut)
{
    return binaryImageFunction<T, addLine<T> >(imIn1, value, imOut);
}

/**
 * Add two images without checking saturation.
 *
 * \param "imIn1 imIn2" Input images.
 * \param imOut Output image.
 * \see add
 */
template <class T>
inline RES_T addNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, addNoSatLine<T> >(imIn1, imIn2, imOut);
}

/**
 * Add a constant value to an image without checking saturation.
 *
 * \param imIn Input image.
 * \param value The constant value to add.
 * \param imOut Output image.
 * \see add
 */
template <class T>
inline RES_T addNoSat(Image<T> &imIn1, const T value, Image<T> &imOut)
{
    return binaryImageFunction<T, addNoSatLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T sub(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, subLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T sub(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, subLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T subNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, subNoSatLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T subNoSat(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, subNoSatLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T sup(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
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
inline RES_T sup(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, supLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T inf(Image<T> &imIn1, T value, Image<T> &imOut)
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
inline RES_T inf(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, infLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T grt(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, grtLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T grt(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, grtLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T grtOrEqu(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, grtOrEquLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T grtOrEqu(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, grtOrEquLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T low(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, lowLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T low(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, lowLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T lowOrEqu(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, lowOrEquLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T lowOrEqu(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, lowLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T div(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, divLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T div(Image<T> &imIn, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, divLine<T> >(imIn, value, imOut);
}

template <class T>
inline RES_T mul(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, mulLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mul(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, mulLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T mulNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, mulLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mulNoSat(Image<T> &imIn1, T value, Image<T> &imOut)
{
    return binaryImageFunction<T, mulLine<T> >(imIn1, value, imOut);
}

template <class T>
inline RES_T logicAnd(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, logicAndLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T logicOr(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, logicOrLine<T> >(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imIn3, Image<T> &imOut)
{
    return tertiaryImageFunction<T, testLine<T> >(imIn1, imIn2, imIn3, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, Image<T> &imIn2, T value, Image<T> &imOut)
{
    return tertiaryImageFunction<T, testLine<T> >(imIn1, imIn2, value, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, T value, Image<T> &imIn2, Image<T> &imOut)
{
    return tertiaryImageFunction<T, testLine<T> >(imIn1, value, imIn2, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn, T value1, T value2, Image<T> &imOut)
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
inline double vol(Image<T> &imIn)
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    int npix = imIn.getPixelCount();
    T *pixels = imIn.getPixels();
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
inline T minVal(Image<T> &imIn)
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    int npix = imIn.getPixelCount();
    T *p = imIn.getPixels();
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
inline T maxVal(Image<T> &imIn)
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    int npix = imIn.getPixelCount();
    T *p = imIn.getPixels();
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
inline RES_T rangeVal(Image<T> &imIn, T *ret_min, T *ret_max)
{
    if (!imIn.isAllocated())
        return RES_ERR;

    int npix = imIn.getPixelCount();
    T *p = imIn.getPixels();
    *ret_min = numeric_limits<T>::max();
    *ret_max = numeric_limits<T>::min();

    for (int i=0;i<npix;i++,p++)
    {
        if (*p<*ret_min)
            *ret_min = *p;
        if (*p>*ret_max)
            *ret_max = *p;
    }

    return RES_OK;
}

/** @}*/

#endif // _D_IMAGE_ARITH_HPP

