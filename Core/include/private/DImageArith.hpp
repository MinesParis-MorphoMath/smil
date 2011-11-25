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


#ifndef _D_IMAGE_ARITH_HPP
#define _D_IMAGE_ARITH_HPP

#include "DLineArith.hpp"

/**
 * \ingroup Core
 * \defgroup Arith
 * @{
 */



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
    unaryImageFunction<T, invLine<T> > iFunc;
    return iFunc(imIn, imOut);
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
    binaryImageFunction<T, addLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
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
    binaryImageFunction<T, addLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
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
    binaryImageFunction<T, addNoSatLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
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
    binaryImageFunction<T, addNoSatLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T sub(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, subLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T sub(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, subLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T subNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, subNoSatLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T subNoSat(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, subNoSatLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T sup(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, supLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
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
    binaryImageFunction<T, supLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T inf(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, infLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
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
    binaryImageFunction<T, infLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T grt(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, grtLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T grt(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, grtLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T grtOrEqu(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, grtOrEquLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T grtOrEqu(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, grtOrEquLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T low(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, lowLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T low(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, lowLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T lowOrEqu(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, lowOrEquLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T lowOrEqu(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, lowLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T div(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, divLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T div(Image<T> &imIn, T value, Image<T> &imOut)
{
    binaryImageFunction<T, divLine<T> > iFunc;
    return iFunc(imIn, value, imOut);
}

template <class T>
inline RES_T mul(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mul(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T mulNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mulNoSat(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T logicAnd(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, logicAndLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T logicOr(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, logicOrLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imIn3, Image<T> &imOut)
{
    tertiaryImageFunction<T, testLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imIn3, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, Image<T> &imIn2, T value, Image<T> &imOut)
{
    tertiaryImageFunction<T, testLine<T> > iFunc;
    return iFunc(imIn1, imIn2, value, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, T value, Image<T> &imIn2, Image<T> &imOut)
{
    tertiaryImageFunction<T, testLine<T> > iFunc;
    return iFunc(imIn1, value, imIn2, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn, T value1, T value2, Image<T> &imOut)
{
    tertiaryImageFunction<T, testLine<T> > iFunc;
    return iFunc(imIn, value1, value2, imOut);
}

template <class T>
inline RES_T fill(Image<T> &imOut, const T value)
{
    if (!areAllocated(&imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;

    typedef typename Image<T>::lineType lineType;
    lineType *lineOut = imOut.getLines();
    int lineLen = imOut.getWidth();
    int lineCount = imOut.getLineCount();

    // Fill first line
//     fillLine<T>::_exec(lineOut[0], lineLen, value);
    fillLine<T> f;
    f(imOut.getPixels(), imOut.getPixelCount(), value);

//     for (int i=1;i<lineCount;i++)
//       memcpy(lineOut[i], lineOut[0], lineLen*sizeof(T));

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
        T1 *pix1 = imIn.getPixels();
        T2 *pix2 = imOut.getPixels();

        int pixCount = imIn.getPixelCount();

        for (int i=0;i<pixCount;i++)
            pix2[i] = static_cast<T2>(pix1[i]);

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
        memcpy(imOut.getPixels(), imIn.getPixels(), imIn.getPixelCount());

        imOut.modified();
        return RES_OK;
    }
    return RES_OK;
}


/** @}*/

#endif // _D_IMAGE_ARITH_HPP

