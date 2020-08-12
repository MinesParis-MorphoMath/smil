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

#ifndef _D_MORPHO_BASE_HPP
#define _D_MORPHO_BASE_HPP

#include "Core/include/DImage.h"
#include "Base/include/private/DImageArith.hpp"
#include "Morpho/include/DMorphoInstance.h"
#include "DMorphImageOperations.hxx"
#include "DHitOrMiss.hpp"

namespace smil
{
  /**
   * @ingroup Morpho
   * @defgroup morpho_base Base Morphological Operators
   * @{
   */

  /**
   * Morphological grayscale dilation
   *
   * @begintheory{dilation}
   * Denoting an image by @f$ f(x) @f$  and the @ref StrElt "structuring
   * function" by @f$ B(x) @f$, the grayscale dilation of @f$ f @f$ by @f$ B @f$
   * is given by @cite serra_image_1982 :
   * @f[ (f\oplus B)(x)=\sup_{y \in \Re^3 }[f(y)+B(x-y)] @f]
   * @endtheory
   *
   * @param[in] imIn Input image
   * @param[out] imOut Output image
   * @param[in] se (optional) The structuring element to use
   * @param[in] borderVal (optional) The border value
   */
  template <class T>
  RES_T dilate(const Image<T> &imIn, Image<T> &imOut,
               const StrElt &se = DEFAULT_SE, T borderVal = ImDtTypes<T>::min())
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    MorphImageFunction<T, supLine<T>> iFunc(borderVal);
    return iFunc(imIn, imOut, se);
  }

  /**
   * Morphological grayscale dilation using the default structuring element but
   * being able to set its size
   *
   * @param[in] imIn Input image
   * @param[out] imOut Output image
   * @param[in] seSize The size of the default structuring element
   * @param[in] borderVal (optional) The border value
   *
   * @note 
   * If you want to use a structuring element different of the default
   * you should set it before
   * @smilexample{example-default-se.py}
   */
  template <class T>
  RES_T dilate(const Image<T> &imIn, Image<T> &imOut, UINT seSize,
               T borderVal = ImDtTypes<T>::min())
  {
    return dilate(imIn, imOut, DEFAULT_SE(seSize), borderVal);
  }

  /**
   * Morphological grayscale erosion
   *
   * @begintheory{erosion}
   * Denoting an image by @f$ f(x) @f$  and the @ref StrElt (or structuring
   * function) by @f$ B(x) @f$, the grayscale erosion of @f$ f @f$ by @f$ B @f$
   * is given by @cite serra_image_1982 :
   * @f[ (f\ominus B)(x)=\inf_{y \in \Re^3 }[f(y)-B(x-y)] @f]
   * @endtheory
   *
   * @param[in] imIn : Input image
   * @param[out] imOut : Output image
   * @param[in] se : Structuring element
   * @param[in] borderVal : (optional) The border value
   */
  template <class T>
  RES_T erode(const Image<T> &imIn, Image<T> &imOut,
              const StrElt &se = DEFAULT_SE, T borderVal = ImDtTypes<T>::max())
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    MorphImageFunction<T, infLine<T>> iFunc(borderVal);
    return iFunc(imIn, imOut, se.transpose());
  }

  /**
   * Morphological grayscale erosion using the default structuring element but
   * being able to set its size
   *
   * @param[in] imIn : Input image
   * @param[out] imOut : Output image
   * @param[in] seSize : The size of the default structuring element
   * @param[in] borderVal : (optional) The border value
   *
   * @note 
   * If you want to use a structuring element different of the default
   * you should set it before
   */
  template <class T>
  RES_T erode(const Image<T> &imIn, Image<T> &imOut, UINT seSize,
              T borderVal = ImDtTypes<T>::max())
  {
    return erode(imIn, imOut, DEFAULT_SE(seSize), borderVal);
  }

  /**
   * Morphological grayscale closing
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T close(const Image<T> &imIn, Image<T> &imOut,
              const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);
    ImageFreezer freeze(imOut);

    bool inplaceSafe = MorphImageFunction<T, supLine<T>>::isInplaceSafe(se);
    Image<T> *imTmp;
    if (inplaceSafe)
      imTmp = &imOut;
    else
      imTmp = new Image<T>(imIn);

    ASSERT((dilate(imIn, *imTmp, se) == RES_OK));
    ASSERT((erode(*imTmp, imOut, se) == RES_OK));

    if (!inplaceSafe)
      delete imTmp;

    return RES_OK;
  }

  /**
   * Morphological grayscale closing using the default structuring element but
   * being able to set its size
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] seSize : size of structuring element
   *
   * @note
   * If you want to use a structuring element different of the default
   * you should set it before
   */
  template <class T>
  RES_T close(const Image<T> &imIn, Image<T> &imOut, UINT seSize)
  {
    return close(imIn, imOut, DEFAULT_SE(seSize));
  }

  /**
   * Morphological grayscale opening
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T open(const Image<T> &imIn, Image<T> &imOut,
             const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);
    ImageFreezer freeze(imOut);

    bool inplaceSafe = MorphImageFunction<T, supLine<T>>::isInplaceSafe(se);
    Image<T> *imTmp;
    if (inplaceSafe)
      imTmp = &imOut;
    else
      imTmp = new Image<T>(imIn);

    ASSERT((erode(imIn, *imTmp, se) == RES_OK));
    ASSERT((dilate(*imTmp, imOut, se) == RES_OK));

    if (!inplaceSafe)
      delete imTmp;

    return RES_OK;
  }

  /**
   * Morphological grayscale opening using the default structuring element but
   * being able to set its size
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] seSize : size of structuring element
   *
   * @note 
   * If you want to use a structuring element different of the default
   * you should set it before
   */
  template <class T>
  RES_T open(const Image<T> &imIn, Image<T> &imOut, UINT seSize)
  {
    return open(imIn, imOut, DEFAULT_SE(seSize));
  }

  /** @} */

} // namespace smil

#endif // _D_MORPHO_BASE_HPP
