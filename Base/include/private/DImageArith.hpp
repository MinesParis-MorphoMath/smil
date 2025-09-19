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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
   * @ingroup Base
   * @defgroup Arith Arithmetic operations
   *
   * @details This modules provides some logic and arithmetic operations on
   * images.
   *
   * @{
   */
  /** @} */

  /**
   * @defgroup  ArithArith Arithmetics
   * @ingroup Arith
   *
   * @addtogroup ArithArith
   *
   * @{
   */
  /**
   * inv() - Invert an image.
   *
   * @param[in] imIn : Input image.
   * @param[out] imOut : Output image.
   *
   * @see Image::operator~
   */
  template <class T>
  RES_T inv(const Image<T> &imIn, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return unaryImageFunction<T, invLine<T>>(imIn, imOut).retVal;
  }

  /**
   * add() - Addition (with saturation check)
   *
   * Addition between two images.
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   * @see Image::operator+
   */
  template <class T>
  RES_T add(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, addLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * add() - Addition (with saturation check)
   *
   * Addition a single value to each pixel of an image.
   *
   * @param[in] imIn1 : input image
   * @param[in] value : value to be added to each pixel in imIn
   * @param[out] imOut : output image
   * @see Image::operator+
   */
  template <class T>
  RES_T add(const Image<T> &imIn1, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imOut);

    return binaryImageFunction<T, addLine<T>>(imIn1, value, imOut);
  }

  /**
   * addNoSat() - Addition (without saturation check)
   *
   * Addition between two images.
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T addNoSat(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, addNoSatLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * addNoSat() - Addition (without saturation check)
   *
   * Addition a single value to each pixel of an image.
   *
   * @param[in] imIn1 : input image
   * @param[in] value : value to be added to each pixel in imIn
   * @param[out] imOut output image
   */
  template <class T>
  RES_T addNoSat(const Image<T> &imIn1, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imOut);

    return binaryImageFunction<T, addNoSatLine<T>>(imIn1, value, imOut);
  }

  /**
   * sub() - Subtraction between two images
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image containing <b>imIn1 - imIn2</b>
   */
  template <class T>
  RES_T sub(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, subLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * sub() - Subtraction between an image and a constant value
   *
   * @param[in] imIn : input image
   * @param[in] value : value to be subtracted from each pixel in the image
   * @param[out] imOut : output image containing <b>imIn - val</b>
   */
  template <class T>
  RES_T sub(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    return binaryImageFunction<T, subLine<T>>(imIn, value, imOut);
  }

  /**
   * sub() - Subtraction between a value and an image
   *
   * @param[in] value : value to which each pixel of the image will be
   * subtracted.
   * @param[in] imIn : input image
   * @param[out] imOut : output image containing <b>val - imIn</b>
   */
  template <class T>
  RES_T sub(const T &value, const Image<T> &imIn, Image<T> &imOut)
  {
    return binaryImageFunction<T, subLine<T>>(value, imIn, imOut);
  }

  /**
   * subNoSat() - Subtraction (without type minimum check) between two images
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image containing <b>imIn1 - imIn2</b>
   */
  template <class T>
  RES_T subNoSat(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, subNoSatLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * subNoSat() - Subtraction (without type minimum check) between an image and
   * a constant value
   *
   * @param[in] imIn : input image
   * @param[in] value : value to be subtracted from each pixel in the image
   * @param[out] imOut : output image containing <b>imIn - val</b>
   */
  template <class T>
  RES_T subNoSat(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, subNoSatLine<T>>(imIn, value, imOut);
  }

  /**
   * subNoSat() - Subtraction (without type minimum check) between a value and
   * an image
   *
   * @param[in] value : value to which each pixel of the image will be
   * subtracted.
   * @param[in] imIn : input image
   * @param[out] imOut : output image containing <b>val - imIn</b>
   */
  template <class T>
  RES_T subNoSat(const T &value, const Image<T> &imIn, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, subNoSatLine<T>>(value, imIn, imOut);
  }

  /**
   * sup() - Sup of two images
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T sup(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, supLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * sup() - Sup of two images
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @return an image, which is the @b sup of @b imIn1 and @b imIn2
   */
  template <class T>
  ResImage<T> sup(const Image<T> &imIn1, const Image<T> &imIn2)
  {
    ResImage<T> newIm(imIn1);

    ASSERT(CHECK_ALLOCATED(&imIn1, &imIn2), newIm);
    ASSERT(CHECK_SAME_SIZE(&imIn1, &imIn2), newIm);

    sup(imIn1, imIn2, newIm);
    return newIm;
  }

  /**
   * sup() - Sup of an image and a value
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T sup(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, supLine<T>>(imIn, value, imOut);
  }

  /**
   * inf() - Inf of two images
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T inf(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, infLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * inf() - Inf of two images
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @return an image, which is the @b inf of @b imIn1 and @b imIn2
   */
  template <class T>
  ResImage<T> inf(const Image<T> &imIn1, const Image<T> &imIn2)
  {
    ResImage<T> newIm(imIn1);

    ASSERT(CHECK_ALLOCATED(&imIn1, &imIn2), newIm);
    ASSERT(CHECK_SAME_SIZE(&imIn1, &imIn2), newIm);

    inf(imIn1, imIn2, newIm);
    return newIm;
  }

  /**
   * inf() - Inf of an image and a value
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T inf(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, infLine<T>>(imIn, value, imOut);
  }

  /**
   * equ() - Equality operator (pixel by pixel)
   *
   * Comparison, pixel by pixel, between two images.
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn1(x) == imIn2(x) ? max(T) : 0)</c></b>
   *
   * The result is an image indicating which pixels are equal in both images.
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   * @note
   * The functions equ() are the inverse of functions diff()
   */
  template <class T>
  RES_T equ(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, equLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * equ() - Equality operator (pixel by pixel)
   *
   * Comparison, pixel by pixel, between an image and a value.
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn(x) == value ? max(T) : 0)</c></b>
   *
   * The result is an image indicating which pixels are equal to the value.
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output image
   *
   * @note
   * The functions equ() are the inverse of functions diff()
   */
  template <class T>
  RES_T equ(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, equLine<T>>(imIn, value, imOut);
  }

  /**
   * equ() - Test equality between two images
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @return @b True if <b>imIn1 == imIn2</b>, @b False otherwise
   *
   * @warning
   * Don't confuse this function with the two others with the same name but
   * checking equality for each pixel. The difference are in the parameters.
   */
  template <class T>
  bool equ(const Image<T> &imIn1, const Image<T> &imIn2)
  {
    ASSERT(CHECK_ALLOCATED(&imIn1, &imIn2), false);
    ASSERT(CHECK_SAME_SIZE(&imIn1, &imIn2), false);

    typedef typename Image<T>::lineType lineType;
    lineType                            pix1 = imIn1.getPixels();
    lineType                            pix2 = imIn2.getPixels();

    for (size_t i = 0; i < imIn1.getPixelCount(); i++)
      if (pix1[i] != pix2[i])
        return false;

    return true;
  }

  /**
   * diff() - Difference between two images.
   *
   * Comparison, pixel by pixel, between two images.
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn1(x) != imIn2(x) ? max(T) : 0)</c></b>
   *
   * In other words, the result is an image indicating which pixels are
   * different in both images.
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   * @note
   * The functions diff() are the inverse of functions equ()
   *
   */
  template <class T>
  RES_T diff(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, diffLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * diff() - Difference between an image and a value
   *
   * Comparison, pixel by pixel, between an image and a value.
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn(x) != value ? max(T) : 0)</c></b>
   *
   * In other words, the result is an image indicating which pixels are
   * different to the value.
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output image
   *
   * @note
   * The functions diff() are the inverse of functions equ()
   */
  template <class T>
  RES_T diff(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, diffLine<T>>(imIn, value, imOut);
  }

  /**
   * absDiff() - Absolute difference ("vertical distance") between two images.
   *
   * Absolute difference between two images : <b><c>abs(imIn1 - imIn2)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T absDiff(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, absDiffLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * grt() - Greater operator
   *
   * Comparison, pixel by pixel, between two images
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn1(x) > imIn2(x) ? max(T) : 0)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T grt(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, grtLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * grt() - Greater operator
   *
   * Comparison, pixel by pixel, between an image and a value.
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn(x) > value ? max(T) : 0)</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T grt(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, grtLine<T>>(imIn, value, imOut);
  }

  /**
   * grtOrEqu() - Greater or equal operator
   *
   * Comparison, pixel by pixel, between two images
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn1(x) >= imIn2(x) ? max(T) : 0)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T grtOrEqu(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, grtOrEquLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * grtOrEqu() - Greater or equal operator
   *
   * Comparison, pixel by pixel, between an image and a value.
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) >= (imIn(x) > value ? max(T) : 0)</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T grtOrEqu(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, grtOrEquLine<T>>(imIn, value, imOut);
  }

  /**
   * low() - Lower operator
   *
   * Comparison, pixel by pixel, between two images
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn1(x) < imIn2(x) ? max(T) : 0)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T low(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, lowLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * low() - Lower operator
   *
   * Comparison, pixel by pixel, between an image and a value.
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn(x) < value ? max(T) : 0)</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T low(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, lowLine<T>>(imIn, value, imOut);
  }

  /**
   * lowOrEqu() - Lower or equal operator
   *
   * Comparison, pixel by pixel, between two images
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn1(x) <= imIn2(x) ? max(T) : 0)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T lowOrEqu(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, lowOrEquLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * lowOrEqu() - Lower or equal operator
   *
   * Comparison, pixel by pixel, between an image and a value
   *
   * The result is a binary image where :
   * - <b><c> imOut(x) = (imIn(x) <= value ? max(T) : 0)</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output imageBase/include/private/DImageArith.hpp:
   *
   */
  template <class T>
  RES_T lowOrEqu(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, lowOrEquLine<T>>(imIn, value, imOut);
  }

  /**
   * div() - Division between two images
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn2(x) != 0 ? imIn1(x) / imIn2(x) : max(T))</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T div(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, divLine<T>>(imIn1, imIn2, imOut);
  }

  /** @cond */
  /*  JOE - 25/08/2020 Is this really needed ??? */
  template <class T>
  RES_T div(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, divLine<T>>(imIn, (double) value, imOut);
  }
  /** @endcond */

  /**
   * div() - Division : an image and a value
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (value != 0 ? imIn(x) / value : max(T))</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] dValue : input value
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T div(const Image<T> &imIn, double &dValue, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    typename ImDtTypes<T>::lineType pixIn  = imIn.getPixels();
    typename ImDtTypes<T>::lineType pixOut = imOut.getPixels();

    for (size_t i = 0; i < imIn.getPixelCount(); i++)
      pixOut[i] = pixIn[i] / dValue;

    return RES_OK;
  }

  /**
   * mul() - Multiply two images
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn1(x) * imIn2(x) <= max(T) ? imIn1(x) * imIn2(x) :
   * max(T))</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T mul(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, mulLine<T>>(imIn1, imIn2, imOut);
  }

  /** @cond */
  /*  JOE - 25/08/2020 Is this really needed ??? */
  template <class T>
  RES_T mul(const Image<T> &imIn1, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imOut);

    return binaryImageFunction<T, mulLine<T>>(imIn1, (double) value, imOut);
  }
  /** @endcond */

  /**
   * mul() - Multiply an image and a value
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn(x) * dValue <= max(T) ? imIn(x) * dValue :
   * max(T))</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] dValue : input value
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T mul(const Image<T> &imIn, const double &dValue, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    typename ImDtTypes<T>::lineType pixIn  = imIn.getPixels();
    typename ImDtTypes<T>::lineType pixOut = imOut.getPixels();
    double                          newVal;

    for (size_t i = 0; i < imIn.getPixelCount(); i++) {
      newVal    = pixIn[i] * dValue;
      pixOut[i] = newVal > double(ImDtTypes<T>::max()) ? ImDtTypes<T>::max()
                                                       : T(newVal);
    }

    return RES_OK;
  }

  /**
   * mulNoSat() - Multiply (without type max check)
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn1(x) * imIn2(x) % (max(T) + 1))</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   * @note In reality mulNoSat() and mul() gives the same result
   */
  template <class T>
  RES_T mulNoSat(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, mulLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * mulNoSat() - Multiply an image and a value (without type max check)
   *
   * The result is a image where the value of each pixel will be given by :
   * - <b><c> imOut(x) = (imIn(x) * value  % (max(T) + 1))</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] value : input value
   * @param[out] imOut : output image
   *
   * @note In reality mulNoSat() and mul() gives the same result
   */
  template <class T>
  RES_T mulNoSat(const Image<T> &imIn, const T &value, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return binaryImageFunction<T, mulLine<T>>(imIn, value, imOut);
  }

  /**
   * log() - Logarithm of an image
   *
   * The result is a image where the value of each pixel will be given by :
   * - <b><c> imOut(x) = (imIn(x) > 0 ? log(imIn(x)) : max(T))</c></b>
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] base : base of the function logarithm
   *
   * @note Possible bases: 0 or none (natural logarithm, or base e), 2, 10
   */
  template <class T1, class T2>
  RES_T log(const Image<T1> &imIn, Image<T2> &imOut, int base = 0)
  {
    ASSERT_ALLOCATED(&imIn);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    unaryImageFunction<T1, logLine<T1, T2>, T2> func;
    func.lineFunction.base = base;
    return func(imIn, imOut);
  }

  /**
   * exp() - exponential of an image
   *
   * The result is a image where the value of each pixel will be given by :
   * - <b><c> imOut(x) = exp(imIn(x))</c></b>
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] base : base of the function logarithm
   *
   * @note
   * - Possible bases: 0 or none (natural logarithm, or base e), 2, 10
   * - Output value is limited to the maximum value of imOut data type
   * @see
   * - log()
   */
  template <class T1, class T2>
  RES_T exp(const Image<T1> &imIn, Image<T2> &imOut, int base = 0)
  {
    ASSERT_ALLOCATED(&imIn);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    unaryImageFunction<T1, expLine<T1, T2>, T2> func;
    func.lineFunction.base = base;
    return func(imIn, imOut);
  }

  /**
   * pow() - power of an image
   *
   * The result is a image where the value of each pixel will be given by :
   * - <b><c> imOut(x) = pow(imIn(x), exponent)</c></b>
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] exponent : exponent to raise the value of each pixel
   *
   * @note
   * - Output value is limited to the maximum value of imOut data type
   */
  template <class T1, class T2>
  RES_T pow(const Image<T1> &imIn, Image<T2> &imOut, double exponent = 2)
  {
    ASSERT_ALLOCATED(&imIn);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    unaryImageFunction<T1, powLine<T1, T2>, T2> func;
    func.lineFunction.exponent = exponent;
    return func(imIn, imOut);
  }

  /**
   * sqrt() - square root of an image
   *
   * The result is a image where the value of each pixel will be given by :
   * - <b><c> imOut(x) = sqrt(imIn(x)</c></b>
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   */
  template <class T1, class T2>
  RES_T sqrt(const Image<T1> &imIn, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    unaryImageFunction<T1, sqrtLine<T1, T2>, T2> func;
    return func(imIn, imOut);
  }

  /** @} */

  /**
   * @defgroup  ArithLogic Logical functions
   * @ingroup Arith
   *
   * @addtogroup ArithLogic
   *
   * @{
   */
  /**
   * logicAnd() - Logic AND operator, pixel by pixel, of two images
   *
   * This function evaluates the logical @b AND of two images. This function
   * works both @b binary and @b  grey images
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn(x) != 0 AND imIn2(x) != 0 ? 1 : 0)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T logicAnd(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, logicAndLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * bitAnd() - Bitwise AND operator, pixel by pixel, of two images
   *
   * This function evaluates the bitwise @b AND of two images. This function
   * works both @b binary and @b  grey images
   *
   * The result is a image where :
   * - <b><c> imOut(x) = imIn(x) & imIn2(x)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T bitAnd(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, bitAndLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * logicOr() - Logic OR operator, pixel by pixel, of two images
   *
   * This function evaluates the logical @b OR of two images. This function
   * works both @b binary and @b  grey images
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn(x) != 0 OR imIn2(x) != 0 ? 1 : 0)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T logicOr(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, logicOrLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * bitOr() - Bitwise OR operator, pixel by pixel, of two images
   *
   * This function evaluates the bitwise @b OR of two images. This function
   * works both @b binary and @b  grey images
   *
   * The result is a image where :
   * - <b><c> imOut(x) = imIn(x) | imIn2(x)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T bitOr(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, bitOrLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * logicXOr() - Logic XOR operator, pixel by pixel, of two images
   *
   * This function evaluates the logical <b>exclusive OR</b> of two images. This
   * function works both @b binary and @b  grey images
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn1(x) != 0 && imIn2(x) == 0) || (imIn1(x) == 0 &&
   * imIn2(x) != 0) ? 1 : 0)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T logicXOr(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, logicXOrLine<T>>(imIn1, imIn2, imOut);
  }

  /**
   * bitXOr() - Bitwise XOR operator, pixel by pixel, of two images
   *
   * This function evaluates the bitwise <b>exclusive OR</b> of two images.
   * This function works both @b binary and @b  grey images
   *
   * The result is a image where :
   * - <b><c> imOut(x) = imIn(x) ^ imIn2(x)</c></b>
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T bitXOr(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2, &imOut);
    ASSERT_SAME_SIZE(&imIn1, &imIn2, &imOut);

    return binaryImageFunction<T, bitXOrLine<T>>(imIn1, imIn2, imOut);
  }
  /** @} */

  /**
   * @defgroup  ArithCompare Comparison functions
   * @ingroup Arith
   *
   * @addtogroup ArithCompare
   *
   * @{
   */
  /**
   * test() - Test
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn(x) != 0 ? imInT(x] : imInF(x))</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] imInT : input image with values to set when @b true
   * @param[in] imInF : input image with values to set when @b false
   * @param[out] imOut : output image
   *
   * @note
   * Can also be used with constant values and result of operators.
   *
   * @par Example
   * @code
   * import smilPython as sp
   * ...
   * sp.test(imIn > 100, 255, 0, imOut)
   * @endcode
   */
  template <class T1, class T2>
  RES_T test(const Image<T1> &imIn, const Image<T2> &imInT,
             const Image<T2> &imInF, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imInT, &imInF, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imInT, &imInF, &imOut);

    return tertiaryImageFunction<T1, testLine<T1, T2>>(imIn, imInT, imInF,
                                                       imOut);
  }

  /**
   * test() - Test
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn(x) != 0 ? imInT(x] : value)</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] imInT : input image with values to set when @b true
   * @param[in] value : input value to set when @b false
   * @param[out] imOut : output image
   *
   */
  template <class T1, class T2>
  RES_T test(const Image<T1> &imIn, const Image<T2> &imInT, const T2 &value,
             Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imInT, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imInT, &imOut);

    return tertiaryImageFunction<T1, testLine<T1, T2>>(imIn, imInT, value,
                                                       imOut);
  }

  /**
   * test() - Test
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn(x) != 0 ? value : imInF(x])</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] value : input value to set when @b true
   * @param[in] imInF : input image with values to set when @b false
   * @param[out] imOut : output image
   *
   */
  template <class T1, class T2>
  RES_T test(const Image<T1> &imIn, const T2 &value, const Image<T2> &imInF,
             Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imInF, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imInF, &imOut);

    return tertiaryImageFunction<T1, testLine<T1, T2>>(imIn, value, imInF,
                                                       imOut);
  }

  /**
   * test() - Test
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imIn(x) != 0 ? value : imInF(x])</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] valueT : input value to set when @b true
   * @param[in] valueF : input value to set when @b false
   * @param[out] imOut : output image
   *
   */
  template <class T1, class T2>
  RES_T test(const Image<T1> &imIn, const T2 &valueT, const T2 &valueF,
             Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    return tertiaryImageFunction<T1, testLine<T1, T2>>(imIn, valueT, valueF,
                                                       imOut);
  }

  /** @cond */
  template <class T1, class imOrValT, class trueT, class falseT, class T2>
  RES_T _compare_base(const Image<T1> &imIn, const char *compareType,
                      const imOrValT &imOrVal, const trueT &trueImOrVal,
                      const falseT &falseImOrVal, Image<T2> &imOut)
  {
    ImageFreezer freeze(imOut);

    Image<T1> tmpIm(imIn);

    if (strcmp(compareType, "==") == 0) {
      ASSERT(equ(imIn, imOrVal, tmpIm) == RES_OK);
    } else if (strcmp(compareType, "!=") == 0) {
      ASSERT(diff(imIn, imOrVal, tmpIm) == RES_OK);
    } else if (strcmp(compareType, ">") == 0) {
      ASSERT(grt(imIn, imOrVal, tmpIm) == RES_OK);
    } else if (strcmp(compareType, "<") == 0) {
      ASSERT(low(imIn, imOrVal, tmpIm) == RES_OK);
    } else if (strcmp(compareType, ">=") == 0) {
      ASSERT(grtOrEqu(imIn, imOrVal, tmpIm) == RES_OK);
    } else if (strcmp(compareType, "<=") == 0) {
      ASSERT(lowOrEqu(imIn, imOrVal, tmpIm) == RES_OK);
    } else {
      ERR_MSG("Unknown operation");
      return RES_ERR;
    }

    ASSERT(test(tmpIm, trueImOrVal, falseImOrVal, imOut) == RES_OK);

    return RES_OK;
  }
  /** @endcond */

  /**
   * compare() - Compare each pixel of two images (or an image and a value)
   *
   * Compare each pixel of two images and set the corresponding pixel of @b
   * imOut to the same pixel of @b trueIm if result is @b true or @b falseIm if
   * not.
   *
   *
   * @param[in] imIn1 : input image
   * @param[in] compareType : one of "<", "<=", "==", "!=", ">=", ">"
   * @param[in] imIn2 : input image (or a constant @b value)
   * @param[in] trueIm : image to take pixel values from when comparison is @b
   * true
   * @param[in] falseIm : image to take pixel values from when comparison is @b
   * false
   * @param[out] imOut : output image
   *
   * @note
   * Parameters of kind image in this call (@b imIn2, @b trueIm and @b falseIm)
   * can be replaced by the respective scalar values (@b value, @b trueVal and
   * @b falseVal). See the next function prototypes.
   */
  template <class T1, class T2>
  RES_T compare(const Image<T1> &imIn1, const char *compareType,
                const Image<T1> &imIn2, const Image<T2> &trueIm,
                const Image<T2> &falseIm, Image<T2> &imOut)
  {
    return _compare_base<T1, Image<T1>, Image<T2>, Image<T2>, T2>(
        imIn1, compareType, imIn2, trueIm, falseIm, imOut);
  }

  /**
   * compare() - Compare each pixel of two images (or an image and a value)
   *
   * Compare each pixel of two images and set the corresponding pixel of @b
   * imOut to @b trueVal if result is @b true or @b falseIm if not.
   *
   * @param[in] imIn1 : input image
   * @param[in] compareType : one of "<", "<=", "==", "!=", ">=", ">"
   * @param[in] imIn2 : input image (or a constant @b value)
   * @param[in] trueVal : value to set when comparison is @b true
   * @param[in] falseIm : image to take pixel values from when comparison is @b
   * false
   * @param[out] imOut : output image
   *
   * @overload
   * @see the first reference to this function
   */
  template <class T1, class T2>
  RES_T compare(const Image<T1> &imIn1, const char *compareType,
                const Image<T1> &imIn2, const T2 &trueVal,
                const Image<T2> &falseIm, Image<T2> &imOut)
  {
    return _compare_base<T1, Image<T1>, T2, Image<T2>, T2>(
        imIn1, compareType, imIn2, trueVal, falseIm, imOut);
  }

  /**
   * compare() - Compare each pixel of two images (or an image and a value)
   *
   * Compare each pixel of two images and set the corresponding pixel of @b
   * imOut to the same pixel of @b trueIm if result is @b true or @b falseVal if
   * not.
   *
   * @param[in] imIn1 : input image
   * @param[in] compareType : one of "<", "<=", "==", "!=", ">=", ">"
   * @param[in] imIn2 : input image
   * @param[in] trueIm : image to take pixel values from when comparison is @b
   * true
   * @param[in] falseVal : value to set when comparison is @b false
   * @param[out] imOut : output image
   *
   * @overload
   * @see the first reference to this function
   */
  template <class T1, class T2>
  RES_T compare(const Image<T1> &imIn1, const char *compareType,
                const Image<T1> &imIn2, const Image<T2> &trueIm,
                const T2 &falseVal, Image<T2> &imOut)
  {
    return _compare_base<T1, Image<T1>, Image<T2>, T2, T2>(
        imIn1, compareType, imIn2, trueIm, falseVal, imOut);
  }

  /**
   * compare() - Compare each pixel of two images (or an image and a value)
   *
   * Compare each pixel of two images and set the corresponding pixel of @b
   * imOut to @b trueVal if result is @b true or @b falseVal if not.
   *
   * @param[in] imIn1 : input image
   * @param[in] compareType : one of "<", "<=", "==", "!=", ">=", ">"
   * @param[in] imIn2 : input image
   * @param[in] trueVal : value to set when comparison is @b true
   * @param[in] falseVal : value to set when comparison is @b false
   * @param[out] imOut : output image
   *
   * @overload
   * @see the first reference to this function
   */
  template <class T1, class T2>
  RES_T compare(const Image<T1> &imIn1, const char *compareType,
                const Image<T1> &imIn2, const T2 &trueVal, const T2 &falseVal,
                Image<T2> &imOut)
  {
    return _compare_base<T1, Image<T1>, T2, T2, T2>(imIn1, compareType, imIn2,
                                                    trueVal, falseVal, imOut);
  }

  /**
   * compare() - Compare each pixel of two images (or an image and a value)
   *
   * Compare each pixel of an image and a value and set the corresponding pixel
   * of @b imOut to the same pixel of @b trueIm if result is @b true or @b
   * falseIm if not.
   *
   * @param[in] imIn : input image
   * @param[in] compareType : one of "<", "<=", "==", "!=", ">=", ">"
   * @param[in] value : input value
   * @param[in] trueIm : image to take pixel values from when comparison is @b
   * true
   * @param[in] falseIm : image to take pixel values from when comparison is @b
   * false
   * @param[out] imOut : output image
   *
   * @overload
   * @see the first reference to this function
   */
  template <class T1, class T2>
  RES_T compare(const Image<T1> &imIn, const char *compareType, const T1 &value,
                const Image<T2> &trueIm, const Image<T2> &falseIm,
                Image<T2> &imOut)
  {
    return _compare_base<T1, T1, Image<T2>, Image<T2>, T2>(
        imIn, compareType, value, trueIm, falseIm, imOut);
  }

  /**
   * compare() - Compare each pixel of two images (or an image and a value)
   *
   * Compare each pixel of an image and a value and set the corresponding pixel
   * of @b imOut to @b trueVal if result is @b true or @b falseIm if not.
   *
   * @param[in] imIn : input image
   * @param[in] compareType : one of "<", "<=", "==", "!=", ">=", ">"
   * @param[in] value : input value
   * @param[in] trueVal : value to set when comparison is @b true
   * @param[in] falseIm : image to take pixel values from when
   * comparison is @b false
   * @param[out] imOut : output image
   *
   * @overload
   * @see the first reference to this function
   */
  template <class T1, class T2>
  RES_T compare(const Image<T1> &imIn, const char *compareType, const T1 &value,
                const T2 &trueVal, const Image<T2> &falseIm, Image<T2> &imOut)
  {
    return _compare_base<T1, T1, T2, Image<T2>, T2>(imIn, compareType, value,
                                                    trueVal, falseIm, imOut);
  }

  /**
   * compare() - Compare each pixel of two images (or an image and a value)
   *
   * Compare each pixel of an image and a value and set the corresponding pixel
   * of @b imOut to the same pixel of @b trueIm if result is @b true or @b
   * falseVal if not.
   *
   * @param[in] imIn : input image
   * @param[in] compareType : one of "<", "<=", "==", "!=", ">=", ">"
   * @param[in] value : input value
   * @param[in] trueIm : image to take pixel values from when comparison is @b
   * true
   * @param[in] falseVal : value to set when comparison is @b false
   * @param[out] imOut : output image
   *
   * @overload
   * @see the first reference to this function
   */
  template <class T1, class T2>
  RES_T compare(const Image<T1> &imIn, const char *compareType, const T1 &value,
                const Image<T2> &trueIm, const T2 &falseVal, Image<T2> &imOut)
  {
    return _compare_base<T1, T1, Image<T2>, T2, T2>(imIn, compareType, value,
                                                    trueIm, falseVal, imOut);
  }

  /**
   * compare() - Compare each pixel of two images (or an image and a value)
   *
   * Compare each pixel of an image and a value and set the corresponding pixel
   * of @b imOut to @b trueVal if result is @b true or @b falseVal if not.
   *
   * @param[in] imIn : input image
   * @param[in] compareType : one of "<", "<=", "==", "!=", ">=", ">"
   * @param[in] value : input value
   * @param[in] trueVal : value to set when comparison is @b true
   * @param[in] falseVal : value to set when comparison is @b false
   * @param[out] imOut : output image
   *
   * @overload
   * @see the first reference to this function
   */
  template <class T1, class T2>
  RES_T compare(const Image<T1> &imIn, const char *compareType, const T1 &value,
                const T2 &trueVal, const T2 &falseVal, Image<T2> &imOut)
  {
    return _compare_base<T1, T1, T2, T2, T2>(imIn, compareType, value, trueVal,
                                             falseVal, imOut);
  }
  /** @} */

  /**
   * @defgroup  ArithRange Value range conversion
   * @ingroup Arith
   *
   * @addtogroup ArithRange
   *
   * @{
   */
  /**

   * fill() - Fill an image with a given value.
   *
   * @param[out] imOut Output image.
   * @param[in] value The value to fill.
   *
   * @vectorized
   * @parallelized
   *
   * @see Image::operator<<
   *
   */
  template <class T>
  RES_T fill(Image<T> &imOut, const T &value)
  {
    ASSERT_ALLOCATED(&imOut);

    return unaryImageFunction<T, fillLine<T>>(imOut, value).retVal;
  }

  /**
   * randFill() - Fill an image with random values.
   *
   * @param[in, out] imOut Output image.
   *
   * @see Image::operator<<
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
    T      minT   = ImDtTypes<T>::min();

    for (size_t i = 0; i < imOut.getPixelCount(); i++)
      pixels[i] = T(rand() / double(RAND_MAX) * rangeT + double(minT));

    imOut.modified();

    return RES_OK;
  }

  /**
   * cast() - Cast from an image type to another
   *
   * Copies the content of @b imIn into @b imOut scaling pixel values to the
   * data type of @b imOut.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   * @note
   * @b imOut shall be a previously allocated image with the same size of @b
   * imIn.
   */
  template <class T1, class T2>
  RES_T cast(const Image<T1> &imIn, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    T1 floor_t1 = ImDtTypes<T1>::min();
    T2 floor_t2 = ImDtTypes<T2>::min();

    // can't use cardinal on floating point data types...
    // double coeff =
    //  double(ImDtTypes<T2>::cardinal()) /  double(ImDtTypes<T1>::cardinal());

    double coeff = double(ImDtTypes<T2>::max() - ImDtTypes<T2>::min()) /
                   double(ImDtTypes<T1>::max() - ImDtTypes<T1>::min());

    typename Image<T1>::lineType pixIn  = imIn.getPixels();
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
      for (i = 0; i < nPix; i++)
        pixOut[i] = floor_t2 + T2(coeff * double(pixIn[i] - floor_t1));
    }

    return RES_OK;
  }

  /**
   * @brief scaleRange() - Linear conversion of pixels values to the range
   * [Min, Max]
   *
   * Values in the input image are linearly mapped into the output image with
   * the following rules :
   * - if <b>imIn(x) <= inMin</b>, imOut(x) will be mapped in the range <b>[0,
   * outMin]</b>
   * - if <b>inMin < imIn(x) <= inMax</b>, imOut(x) will be mapped in the range
   * <b>[outMin, outMax]</b>
   * - if <b>imIn(x) > inMax</b>, imOut(x) will be mapped in the range
   * <b>[outMax, max(T2)]</b>
   *
   * @param[in] imIn : input Image
   * @param[in] inMin, inMax : control range in the input image
   * @param[in] outMin, outMax : control range in the output image
   * @param[out] imOut : output Image
   *
   */
  template <class T1, class T2>
  RES_T scaleRange(const Image<T1> &imIn, const T1 inMin, const T1 inMax,
                   const T2 outMin, const T2 outMax, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    size_t W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    if ((inMax - inMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    T1 inTop  = imIn.getDataTypeMax();
    T2 outTop = imOut.getDataTypeMax();

    double k1, k2, k3;
    k1 = k2 = k3 = 0;

    if (inMin > 0)
      k1 = ((double) (outMin - 0)) / ((double) (inMin - 0));
    if (inMax > inMin)
      k2 = ((double) (outMax - outMin)) / ((double) (inMax - inMin));
    if (inTop > inMax)
      k3 = ((double) (outTop - outMax)) / ((double) (inTop - inMax));

    size_t iMax = W * H * D;
    for (size_t i = 0; i < iMax; i++) {
      if (bufferIn[i] < inMin) {
        bufferOut[i] = (T2)(k1 * bufferIn[i]);
        continue;
      }
      if (bufferIn[i] >= inMin && bufferIn[i] < inMax) {
        bufferOut[i] = (T2)(outMin + k2 * (bufferIn[i] - inMin));
        continue;
      }
      if (bufferIn[i] >= inMax) {
        bufferOut[i] = (T2)(outMax + k3 * (bufferIn[i] - inMax));
      }
    }
    return RES_OK;
  }

  /**
   * @brief scaleRange() - Linear conversion of pixel values to the range
   * [Min, Max]
   *
   * Maps a range of values in the input image into the range <b>[Min, Max]</b>
   * in the output image.
   *
   * If @b onlyNonZero is @b true uses <b>[minVal(), maxVal()]</b> as the range
   * of values in the input image, otherwise, uses the full range of values.
   *
   * @param[in] imIn : input image
   * @param[in] Min : Minimum value in the output image
   * @param[in] Max : Maximum value in the output image
   * @param[out] imOut : output Image
   * @param[in] onlyNonZero : defines how to find input image range of values
   */
  template <class T1, class T2>
  RES_T scaleRange(const Image<T1> &imIn, const T2 Min, const T2 Max,
                   Image<T2> &imOut, bool onlyNonZero)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    size_t W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    T1 vMin, vMax;
    if (onlyNonZero) {
      vMin = minVal(imIn);
      vMax = maxVal(imIn);
    } else {
      vMin = imIn.getDataTypeMin();
      vMax = imIn.getDataTypeMax();
    }
    if ((vMax - vMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    double k = ((double) (Max - Min)) / ((double) (vMax - vMin));

    size_t iMax = W * H * D;
    for (size_t i = 0; i < iMax; i++)
      bufferOut[i] = (T2)(Min + k * (bufferIn[i] - vMin));

    return RES_OK;
  }

  /**
   * @brief scaleRange() - Linear conversion of pixels values to the
   * domain range
   *
   * Maps a range in the input image into the  range <b>[min(T2), max(T2)]</b>
   * in the output image.
   *
   * If @b onlyNonZero is @b true uses <b>[minVal(), maxVal()]</b> as the range
   * of values in the input image, otherwise, uses the full range of values.
   *
   * @param[in] imIn : input Image
   * @param[out] imOut : output Image
   * @param[in] onlyNonZero : defines how to find input image range of values
   */
  template <class T1, class T2>
  RES_T scaleRange(const Image<T1> &imIn, Image<T2> &imOut, bool onlyNonZero)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

#if 1
    return scaleRange(imIn, imOut.getDataTypeMin(), imOut.getDataTypeMax(),
                      imOut, onlyNonZero);
#else

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    size_t W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    T2 Max = imOut.getDataTypeMax();
    T2 Min = imOut.getDataTypeMin();

    T1 vMin, vMax;
    if (onlyNonZero) {
      vMin = minVal(imIn);
      vMax = maxVal(imIn);
    } else {
      vMin = imIn.getDataTypeMin();
      vMax = imIn.getDataTypeMax();
    }

    if ((vMax - vMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    double k = ((double) (Max - Min)) / ((double) (vMax - vMin));

    size_t iMax = W * H * D;
    for (size_t i = 0; i < iMax; i++)
      bufferOut[i] = (T2)(Min + k * (bufferIn[i] - vMin));

    return RES_OK;
#endif
  }

  /**
   * @brief sCurve() - S Curve transform
   *
   * This function emulates the <b>S Curve</b> caracteristic of film
   * photography.
   *
   * Use a sigmoid function centered at @b pivot with derivative @b ratio.
   *
   * One use of this filter is to set (increase or decrease) the contrast in the
   * neighborhood of the @b pivot.
   *
   * @param[in] imIn : input Image
   * @param[in] pivot :
   * * if 0, takes the median of the histogram of input image as pivot
   * * otherwise, use this value
   * @param[in] ratio : derivative of output image at pivot value
   * @param[out] imOut : output Image
   */
  template <class T1, class T2>
  RES_T sCurve(const Image<T1> &imIn, const T1 pivot, const double ratio,
               Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    size_t W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    T1 vMin = minVal(imIn);
    T1 vMax = maxVal(imIn);
    if ((vMax - vMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    T1 ctr = pivot;
    if (pivot == 0 || pivot > vMax)
      ctr = (vMax - vMin) / 2;

    double k = 4. * ratio / (vMax - vMin);

    T2     Max  = imOut.getDataTypeMax();
    size_t iMax = W * H * D;

    for (size_t i = 0; i < iMax; i++)
      bufferOut[i] = (T2)(Max / (1. + std::exp(-k * (bufferIn[i] - ctr))));

    return RES_OK;
  }

  /** @} */

  /**
   * @defgroup  ArithChannel Operations on image channels
   * @ingroup Arith
   *
   * @addtogroup ArithChannel
   *
   * @{
   */
  /**
   * copyChannel() - Copy a channel of multichannel image into a single channel
   * image
   *
   * @param[in] imIn : input image
   * @param[in] chanNum : channel in the input image to copy
   * @param[out] imOut : output image
   *
   * @note
   * - @b imIn et @b imOut are @b 2D images
   * - @b imOut shall be a previosly allocated single channel image with the
   * same size than @b imIn.
   * - @b chanNum shall be a valid channel of @b imIn.
   *
   * @smilexample{multichannel_operations.py}
   */
  template <class MCT1, class T2>
  RES_T copyChannel(const Image<MCT1> &imIn, const UINT &chanNum,
                    Image<T2> &imOut)
  {
    ASSERT(chanNum < MCT1::channelNumber());
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    typedef typename MCT1::DataType T1;
    typename Image<T1>::lineType    lineIn  = imIn.getPixels().arrays[chanNum];
    typename Image<T2>::lineType    lineOut = imOut.getPixels();

    copyLine<T1, T2>(lineIn, imIn.getPixelCount(), lineOut);
    imOut.modified();
    return RES_OK;
  }

  /**
   * copyToChannel() - Copy a single channel image into a channel of
   * multichannel image
   *
   * @param[in] imIn : input image
   * @param[in] chanNum : channel in the output image to copy
   * @param[out] imOut : output image
   *
   * @note
   * - @b imIn et @b imOut are @b 2D images
   * - @b imOut shall be a previosly allocated multichannel image with the same
   * size than @b imIn.
   * - @b chanNum shall be a valid channel of @b imOut.
   *
   * @smilexample{multichannel_operations.py}
   */
  template <class T1, class MCT2>
  RES_T copyToChannel(const Image<T1> &imIn, const UINT &chanNum,
                      Image<MCT2> &imOut)
  {
    ASSERT(chanNum < MCT2::channelNumber());
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    typedef typename MCT2::DataType T2;
    typename Image<T1>::lineType    lineIn  = imIn.getPixels();
    typename Image<T2>::lineType    lineOut = imOut.getPixels().arrays[chanNum];

    copyLine<T1, T2>(lineIn, imIn.getPixelCount(), lineOut);
    imOut.modified();
    return RES_OK;
  }

  /**
   * splitChannels() - Split channels of multichannel image to a 3D image with
   * each channel on a Z slice
   *
   * @param[in] imIn : input image
   * @param[out] im3DOut : output image
   *
   * @note
   * - @b imIn is a @b 2D image
   * - @b im3DOut whall be a previously allocated image. Its size will be set to
   * the same as @b imIn, but its depth will be set the the number of channels
   * of @b imIn.
   *
   * @smilexample{multichannel_operations.py}
   */
  template <class MCT1, class T2>
  RES_T splitChannels(const Image<MCT1> &imIn, Image<T2> &im3DOut)
  {
    ASSERT_ALLOCATED(&imIn);

    UINT width = imIn.getWidth(), height = imIn.getHeight();
    UINT chanNum  = MCT1::channelNumber();
    UINT pixCount = width * height;
    ASSERT(im3DOut.setSize(width, height, chanNum) == RES_OK);

    typedef typename MCT1::DataType T1;
    typename Image<MCT1>::lineType  lineIn  = imIn.getPixels();
    typename Image<T2>::lineType    lineOut = im3DOut.getPixels();

    for (UINT i = 0; i < chanNum; i++) {
      copyLine<T1, T2>(lineIn.arrays[i], pixCount, lineOut);
      lineOut += pixCount;
    }
    im3DOut.modified();

    return RES_OK;
  }

  /**
   * mergeChannels() - Merge slices of a 3D image into a multichannel image
   *
   * This function has the inverse behaviour of function splitChannels()
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   * @smilexample{multichannel_operations.py}
   */
  template <class T1, class MCT2>
  RES_T mergeChannels(const Image<T1> &imIn, Image<MCT2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn);
    UINT chanNum = MCT2::channelNumber();
    ASSERT(imIn.getDepth() == chanNum);

    UINT width = imIn.getWidth(), height = imIn.getHeight();
    UINT pixCount = width * height;
    imOut.setSize(width, height);

    typedef typename MCT2::DataType T2;
    typename Image<T1>::lineType    lineIn  = imIn.getPixels();
    typename Image<MCT2>::lineType  lineOut = imOut.getPixels();

    for (UINT i = 0; i < chanNum; i++) {
      copyLine<T1, T2>(lineIn, pixCount, lineOut.arrays[i]);
      lineIn += pixCount;
    }
    imOut.modified();

    return RES_OK;
  }
  /** @} */

  /**
   * @defgroup  ArithOthers Others functions
   * @ingroup Arith
   *
   * @addtogroup ArithOthers
   *
   * @{
   */
  /**
   * mask() - Image mask
   *
   * The result is a image where :
   * - <b><c> imOut(x) = (imMask(x) != 0 ? imIn(x) : 0)</c></b>
   *
   * @param[in] imIn : input image
   * @param[in] imMask : input mask image
   * @param[out] imOut : output image
   *
   */
  template <class T>
  RES_T mask(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut)
  {
    return test<T>(imMask, imIn, T(0), imOut);
  }

  /**
   * applyLookup() - Apply a lookup map to a labeled image
   *
   * Converts a labeled image into a labeled image, converting labels based on a
   * lookup map (or dictionary, in @b python)
   *
   * @param[in] imIn : labeled input image
   * @param[in] _map : lookup map
   * @param[out] imOut : output labeled image
   * @param[in] defaultValue : values to be assigned when the input value isn't
   * present in the keys of the lookup map <b>(_map)</b>
   *
   * @smilexample{example-applylookup.py}
   *
   */
  template <class T1, class mapT, class T2>
  RES_T applyLookup(const Image<T1> &imIn, const mapT &_map, Image<T2> &imOut,
                    T2 defaultValue = T2(0))
  {
    ASSERT(!_map.empty(), "Input map is empty", RES_ERR);
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    // Verify that the max(measure) doesn't exceed the T2 type max
    typename mapT::const_iterator max_it =
        std::max_element(_map.begin(), _map.end(), map_comp_value_less());
    ASSERT((max_it->second <= ImDtTypes<T2>::max()),
           "Input map max exceeds data type max!", RES_ERR);

    typename Image<T1>::lineType pixIn  = imIn.getPixels();
    typename Image<T2>::lineType pixOut = imOut.getPixels();

    typename mapT::const_iterator it;

    for (size_t i = 0; i < imIn.getPixelCount(); i++) {
      it = _map.find(*pixIn);
      if (it != _map.end())
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
  /** @cond */
  template <class T1, class T2>
  // SFINAE General case
  ENABLE_IF(!IS_SAME(T1, UINT8) && !IS_SAME(T1, UINT16), RES_T)
  applyLookup(const Image<T1> &imIn, const std::map<T1, T2> &lut,
                  Image<T2> &imOut, T2 defaultValue = T2(0))
  {
    return applyLookup<T1, std::map<T1, T2>, T2>(imIn, lut, imOut, defaultValue);
  }

  // Specialization for T1 == UINT8 or T1 == UINT16
  template <class T1, class T2>
  // SFINAE For T1 == UINT8 || T1 == UINT16
  ENABLE_IF(IS_SAME(T1, UINT8) || IS_SAME(T1, UINT16), RES_T)
    applyLookup(const Image<T1> &imIn, const std::map<T1, T2> &lut,
                  Image<T2> &imOut, T2 defaultValue = T2(0))
  {
    ASSERT(!lut.empty(), "Input map is empty", RES_ERR);
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    T2 *outVals = ImDtTypes<T2>::createLine(ImDtTypes<T1>::cardinal());

    for (int i = 0; i < ImDtTypes<T1>::max(); i++)
      outVals[i] = defaultValue;

    typename Image<T1>::lineType pixIn  = imIn.getPixels();
    typename Image<T2>::lineType pixOut = imOut.getPixels();

    for (typename std::map<T1, T2>::const_iterator it = lut.begin(); it != lut.end();
         it++)
      outVals[it->first] = it->second;

    for (size_t i = 0; i < imIn.getPixelCount(); i++)
      pixOut[i] = outVals[pixIn[i]];

    imOut.modified();

    ImDtTypes<T2>::deleteLine(outVals);

    return RES_OK;
  }
  /** @endcond */
#else  // SWIG
  template <class T1, class T2>
  RES_T applyLookup(const Image<T1> &imIn, const map<T1, T2> &lut,
                    Image<T2> &imOut, T2 defaultValue = T2(0));
#endif // SWIG

  /** @}*/

} // namespace smil

#endif // _D_IMAGE_ARITH_HPP
