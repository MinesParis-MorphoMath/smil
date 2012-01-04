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


#ifndef _D_IMAGE_ARITH_BIN_HPP
#define _D_IMAGE_ARITH_BIN_HPP

#include "DBinary.hpp"
#include "DLineArith_BIN.hxx"

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
template <>
inline double vol<bool>(Image<bool> &imIn)
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    UINT lineLen = imIn.getWidth();
    BIN::lineType *cur_line = (BIN::lineType*)imIn.getLines();
    double vol = 0;
    
    for (int j=0;j<imIn.getLineCount();j++,cur_line++)
    {
	BIN_TYPE *pix = *cur_line;
	for (int i=0;i<lineLen;i++)
	  if (pix[i/BIN::SIZE] & (1UL<<(i%BIN::SIZE)))
	    vol += 1;
    }

    return vol;
}

/**
 * Min value of an image
 *
 * Returns the min of the pixel values.
 * \param imIn Input image.
 */
// template <class T>
// inline T minVal(Image<T> &imIn)
// {
//     if (!imIn.isAllocated())
//         return RES_ERR_BAD_ALLOCATION;
// 
//     int npix = imIn.getPixelCount();
//     T *p = imIn.getPixels();
//     T minVal = numeric_limits<T>::max();
// 
//     for (int i=0;i<npix;i++,p++)
//         if (*p<minVal)
//             minVal = *p;
// 
//     return minVal;
// }

/**
 * Max value of an image
 *
 * Returns the min of the pixel values.
 * \param imIn Input image.
 */
// template <class T>
// inline T maxVal(Image<T> &imIn)
// {
//     if (!imIn.isAllocated())
//         return RES_ERR_BAD_ALLOCATION;
// 
//     int npix = imIn.getPixelCount();
//     T *p = imIn.getPixels();
//     T maxVal = numeric_limits<T>::min();
// 
//     for (int i=0;i<npix;i++,p++)
//         if (*p>maxVal)
//             maxVal = *p;
// 
//     return maxVal;
// }

/**
 * Min and Max values of an image
 *
 * Returns the min and the max of the pixel values.
 * \param imIn Input image.
 */
// template <class T>
// inline RES_T rangeVal(Image<T> &imIn, T *ret_min, T *ret_max)
// {
//     if (!imIn.isAllocated())
//         return RES_ERR;
// 
//     int npix = imIn.getPixelCount();
//     T *p = imIn.getPixels();
//     *ret_min = numeric_limits<T>::max();
//     *ret_max = numeric_limits<T>::min();
// 
//     for (int i=0;i<npix;i++,p++)
//     {
//         if (*p<*ret_min)
//             *ret_min = *p;
//         if (*p>*ret_max)
//             *ret_max = *p;
//     }
// 
//     return RES_OK;
// }

// template <>
// inline bool equ(Image<bool> &imIn1, Image<bool> &imIn2)
// {
//     if (!imIn1.isAllocated())
//         return RES_ERR_BAD_ALLOCATION;
// 
//     UINT lineLen = imIn1.getWidth();
//     BIN::lineType *line1 = (BIN::lineType*)imIn1.getLines();
//     BIN::lineType *line2 = (BIN::lineType*)imIn2.getLines();
//     double vol = 0;
//     
//     for (int j=0;j<imIn1.getLineCount();j++,line1++,line2++)
//     {
// 	BIN_TYPE *pix1 = *line1;
// 	BIN_TYPE *pix2 = *line2;
// 	for (int i=0;i<lineLen;i++)
// 	  if (pix1[i/BIN::SIZE] & (1UL<<(i%BIN::SIZE)) != pix2[i/BIN::SIZE] & (1UL<<(i%BIN::SIZE)))
// 	    return false;
//     }
// 
//     return true;
// }

/** @}*/

#endif // _D_IMAGE_ARITH_BIN_HPP

