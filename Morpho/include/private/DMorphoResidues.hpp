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

#ifndef _D_MORPHO_RESIDUES_HPP
#define _D_MORPHO_RESIDUES_HPP

#include "DMorphoBase.hpp"

namespace smil
{
  /**
   * @ingroup Morpho
   * @defgroup Residues Morphological Residues
   *
   * @details In Mathematical morphology, a @TB{Residue} is the symetrical
   * difference between an image and some transformation or between two
   * transformations.
   *
   * @{
   */

  /**
   * gradient() - Morphological gradient
   *
   * @details : The morphological gradient of an image is defined as the
   * difference between its @b dilation and its @b erosion :
   * @f[ gradient(im) = \epsilon(im) - \delta(im) @f]
   *
   * @see @SoilleBook{p. 85-89, 127-130}
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T gradient(const Image<T> &imIn, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    return gradient(imIn, imOut, se, se);
  }

  /**
   * gradient() - Morphological gradient
   *
   * @details : The morphological gradient of an image is defined as the
   * difference between its @b dilation and its @b erosion :
   * @f[ gradient(im) = \epsilon(im) - \delta(im) @f]
   *
   * This function allows the use of different structuring elements for the
   * @b dilation and @b erosion.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] dilSe : @b dilation structuring element
   * @param[in] eroSe : @b erosion structuring element
   *
   * @overload
   */
  template <class T>
  RES_T gradient(const Image<T> &imIn, Image<T> &imOut, const StrElt &dilSe,
                 const StrElt &eroSe)
  {
    Image<T> dilIm(imIn);
    Image<T> eroIm(imIn);

    RES_T res = dilate(imIn, dilIm, dilSe);
    if (res == RES_OK)
      res = erode(imIn, eroIm, eroSe);
    if (res == RES_OK)
      res = sub(dilIm, eroIm, imOut);
    return res;
  }


  /**
   * topHat() - Top-Hat
   *
   * @TB{Top-Hat} or @TB{Open top-hat} or @TB{White top-hat} is
   * defined as the difference of  the image and its @b opening : 
   * @f[ WTH(im) = im - \gamma(im) @f]
   *
   * @see @SoilleBook{p. 121-127}
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T topHat(const Image<T> &imIn, Image<T> &imOut,
               const StrElt &se = DEFAULT_SE)
  {
    Image<T> openIm(imIn);

    RES_T res = open(imIn, openIm, se);
    if (res == RES_OK)
      res = sub(imIn, openIm, imOut);
    return res;
  }

  /**
   * dualTopHat() - Dual Top-Hat
   *
   * @TB{Dual Top-Hat} or @TB{Close top-hat} or
   * @TB{Black top-hat} is defined as the difference of @b closing of the
   * image and itself :  @f[ BTH(im) = \phi(im) - im @f]
   *
   * @see @SoilleBook{p. 121-127}
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T dualTopHat(const Image<T> &imIn, Image<T> &imOut,
                   const StrElt &se = DEFAULT_SE)
  {
    Image<T> closeIm(imIn);

    RES_T res = close(imIn, closeIm, se);
    if (res == RES_OK)
      res = sub(closeIm, imIn, imOut);
    return res;
  }

  /** @}*/

} // namespace smil

#endif // _D_MORPHO_RESIDUES_HPP
