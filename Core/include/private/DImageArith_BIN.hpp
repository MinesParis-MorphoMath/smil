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
// template <class T>
// inline double vol(Image<T> &imIn)
// {
//     if (!imIn.isAllocated())
//         return RES_ERR_BAD_ALLOCATION;
// 
//     int npix = imIn.getPixelCount();
//     T *pixels = imIn.getPixels();
//     double vol = 0;
// 
//     for (int i=0;i<npix;i++)
//         vol += pixels[i];
// 
//     return vol;
// }

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

/** @}*/

#endif // _D_IMAGE_ARITH_BIN_HPP

