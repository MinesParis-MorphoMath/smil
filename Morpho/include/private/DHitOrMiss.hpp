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

#ifndef _D_THINNING_HPP
#define _D_THINNING_HPP

#include "Morpho/include/DCompositeSE.h"
#include "Morpho/include/private/DMorphoBase.hpp"

namespace smil
{
  /**
   * @ingroup Morpho
   * @defgroup HitOrMiss Hit-or-Miss Transform
   *
   * @anchor ops-hit-or-miss
   * @brief Hit-or-miss Morphological Operations
   *
   * In mathematical morphology, hit-or-miss transform is an operation that
   * detects a given configuration (or pattern) in a binary image, using the
   * morphological erosion operator and a pair of disjoint structuring elements
   * (see CompStrElt and CompStrEltList).
   * The result of the hit-or-miss transform is the set of positions where the
   * first structuring element fits in the foreground of the input image, and
   * the second structuring element misses it completely.
   *
   * @see 
   * - @soillebook{p. 139}
   * - <a href="https://en.wikipedia.org/wiki/Hit-or-miss_transform">
   * Hit-or-mis Transform</a>
   *
   * @{
   */

  /**
   * hitOrMiss() - Hit Or Miss transform
   *
   * @param[in] imIn : input image
   * @param[in] foreSE, backSE : foreground an background structuring elements
   * @param[out] imOut : output image
   * @param[in] borderVal : value to be assigned to border pixels
   */
  template <class T>
  RES_T hitOrMiss(const Image<T> &imIn, const StrElt &foreSE,
                  const StrElt &backSE, Image<T> &imOut,
                  T borderVal = ImDtTypes<T>::min())
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    Image<T> tmpIm(imIn);
    ASSERT_ALLOCATED(&tmpIm);
    ImageFreezer freezer(imOut);
    ASSERT((inv<T>(imIn, tmpIm) == RES_OK));
    ASSERT((erode(tmpIm, imOut, backSE, borderVal) == RES_OK));
    ASSERT((erode(imIn, tmpIm, foreSE, borderVal) == RES_OK));
    ASSERT((inf(tmpIm, imOut, imOut) == RES_OK));

    return RES_OK;
  }

  /**
   * hitOrMiss() - Hit Or Miss transform
   *
   * @param[in] imIn : input image
   * @param[in] compSE : composite structuring element with both foreground and
   * background structuring elements
   * @param[out] imOut : output image
   * @param[in] borderVal : value to be assigned to border pixels
   *
   * @overload
   */
  template <class T>
  RES_T hitOrMiss(const Image<T> &imIn, const CompStrElt &compSE,
                  Image<T> &imOut, T borderVal = ImDtTypes<T>::min())
  {
    return hitOrMiss(imIn, compSE.fgSE, compSE.bgSE, imOut, borderVal);
  }

  /**
   * hitOrMiss() - Hit Or Miss transform
   *
   * @param[in] imIn : input image
   * @param[in] mhtSE : vector with composite structuring elements with both
   * foreground and background structuring elements
   * @param[out] imOut : output image
   * @param[in] borderVal : value to be assigned to border pixels
   *
   * @overload
   */
  template <class T>
  RES_T hitOrMiss(const Image<T> &imIn, const CompStrEltList &mhtSE,
                  Image<T> &imOut, T borderVal = ImDtTypes<T>::min())
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    Image<T> tmpIm(imIn);
    ASSERT_ALLOCATED(&tmpIm);

    ImageFreezer freezer(imOut);
    ASSERT((fill(imOut, ImDtTypes<T>::min()) == RES_OK));
    for (std::vector<CompStrElt>::const_iterator it = mhtSE.compSeList.begin();
         it != mhtSE.compSeList.end(); it++) {
      ASSERT((hitOrMiss<T>(imIn, (*it).fgSE, (*it).bgSE, tmpIm, borderVal) ==
              RES_OK));
      ASSERT((sup(imOut, tmpIm, imOut) == RES_OK));
    }

    return RES_OK;
  }

  /**
   * thin() - Thinning transform
   *
   * @b Thinnings consist in removing foreground image pixels matching a
   * configuration given by a composite SE. In other words, the hit-or-miss
   * transform of the image by this SE is subtracted from the original image.
   *
   * @param[in] imIn : input image
   * @param[in] foreSE, backSE : foreground an background structuring elements
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T thin(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE,
             Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    Image<T> tmpIm(imIn);
    ASSERT_ALLOCATED(&tmpIm);
    ImageFreezer freezer(imOut);

    ASSERT((hitOrMiss(imIn, foreSE, backSE, tmpIm) == RES_OK));
    ASSERT((inv(tmpIm, tmpIm) == RES_OK));
    ASSERT((inf(imIn, tmpIm, imOut) == RES_OK));

    return RES_OK;
  }

  /**
   * thin() - Thinning transform
   *
   * @param[in] imIn : input image
   * @param[in] compSE : composite structuring element with both foreground and
   * background structuring elements
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T thin(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut)
  {
    return thin(imIn, compSE.fgSE, compSE.bgSE, imOut);
  }

  /**
   * thin() - Thinning transform
   *
   * @param[in] imIn : input image
   * @param[in] mhtSE : vector with composite structuring elements with both
   * foreground and background structuring elements
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T thin(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    Image<T> tmpIm(imIn, true); // clone
    ASSERT_ALLOCATED(&tmpIm);

    ImageFreezer freezer(imOut);
    ASSERT((fill(imOut, ImDtTypes<T>::min()) == RES_OK));
    for (std::vector<CompStrElt>::const_iterator it = mhtSE.compSeList.begin();
         it != mhtSE.compSeList.end(); it++) {
      ASSERT((thin<T>(tmpIm, (*it).fgSE, (*it).bgSE, tmpIm) == RES_OK));
    }
    copy(tmpIm, imOut);

    return RES_OK;
  }

  /**
   * thick() - Thicking transform
   *
   * A @b thickening consists in adding background pixels having a specific
   * configuration to the set of foreground pixels. This is achieved by adding 
   * to the input image the hit or miss transform by the corresponding composite
   * SE.
   *
   * @param[in] imIn : input image
   * @param[in] foreSE, backSE : foreground an background structuring elements
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T thick(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE,
              Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    Image<T> tmpIm(imIn);
    ASSERT_ALLOCATED(&tmpIm);
    ImageFreezer freezer(imOut);

    ASSERT((hitOrMiss(imIn, foreSE, backSE, tmpIm) == RES_OK));
    ASSERT((sup(imIn, tmpIm, imOut) == RES_OK));

    return RES_OK;
  }

  /**
   * thick() - Thicking transform
   *
   * @param[in] imIn : input image
   * @param[in] compSE : composite structuring element with both foreground and
   * background structuring elements
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T thick(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut)
  {
    return thick(imIn, compSE.fgSE, compSE.bgSE, imOut);
  }

  /**
   * thick() - Thicking transform
   *
   * @param[in] imIn : input image
   * @param[in] mhtSE : vector with composite structuring elements with both
   * foreground and background structuring elements
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T thick(const Image<T> &imIn, const CompStrEltList &mhtSE,
              Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    Image<T> tmpIm(imIn, true); // clone
    ASSERT_ALLOCATED(&tmpIm);

    ImageFreezer freezer(imOut);
    ASSERT((fill(imOut, ImDtTypes<T>::min()) == RES_OK));
    for (std::vector<CompStrElt>::const_iterator it = mhtSE.compSeList.begin();
         it != mhtSE.compSeList.end(); it++) {
      ASSERT((thick<T>(tmpIm, (*it).fgSE, (*it).bgSE, tmpIm) == RES_OK));
    }
    copy(tmpIm, imOut);

    return RES_OK;
  }

  /**
   * fullThin() - Thinning transform (full)
   *
   * thin() applied repeatedly till output image remains "stable". Output image
   * "stability" is defined when the volume of the output image remains stops
   * changing (@txtitalic{idempotence}).
   *
   * @param[in] imIn : input image
   * @param[in] mhtSE : vector with composite structuring elements with both
   * foreground and background structuring elements
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T fullThin(const Image<T> &imIn, const CompStrEltList &mhtSE,
                 Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freezer(imOut);

    double v1, v2;
    ASSERT((thin<T>(imIn, mhtSE, imOut) == RES_OK));
    v1 = vol(imOut);
    while (true) {
      ASSERT((thin<T>(imOut, mhtSE, imOut) == RES_OK));
      v2 = vol(imOut);
      if (v2 == v1)
        break;
      v1 = v2;
    }

    return RES_OK;
  }

  /**
   * fullThin() - Thinning transform (full)
   *
   * @param[in] imIn : input image
   * @param[in] compSE : composite structuring element with both foreground and
   * background structuring elements
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T fullThin(const Image<T> &imIn, const CompStrElt &compSE,
                 Image<T> &imOut)
  {
    return fullThin(imIn, CompStrEltList(compSE), imOut);
  }

  /**
   * fullThin() - Thinning transform (full)
   *
   * @param[in] imIn : input image
   * @param[in] foreSE, backSE : foreground an background structuring elements
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T fullThin(const Image<T> &imIn, const StrElt &foreSE,
                 const StrElt &backSE, Image<T> &imOut)
  {
    return fullThin(imIn, CompStrEltList(CompStrElt(foreSE, backSE)), imOut);
  }

  /**
   * fullThick() - Thicking transform (full)
   *
   * thick() applied repeatedly till output image remains "stable". Output image
   * "stability" is defined when the volume of the output image remains stops
   * changing (@txtitalic{idempotence}).
   *
   * @param[in] imIn : input image
   * @param[in] mhtSE : vector with composite structuring elements with both
   * foreground and background structuring elements
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T fullThick(const Image<T> &imIn, const CompStrEltList &mhtSE,
                  Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freezer(imOut);
    double v1, v2;
    ASSERT((thick<T>(imIn, mhtSE, imOut) == RES_OK));
    v1 = vol(imOut);
    while (true) {
      ASSERT((thick<T>(imOut, mhtSE, imOut) == RES_OK));
      v2 = vol(imOut);
      if (v2 == v1)
        break;
      v1 = v2;
    }

    return RES_OK;
  }

  /**
   * fullThick() - Thicking transform (full)
   *
   * @param[in] imIn : input image
   * @param[in] compSE : composite structuring element with both foreground and
   * background structuring elements
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T fullThick(const Image<T> &imIn, const CompStrElt &compSE,
                  Image<T> &imOut)
  {
    return fullThick(imIn, CompStrEltList(compSE), imOut);
  }

  /**
   * fullThick() - Thicking transform (full)
   *
   * @param[in] imIn : input image
   * @param[in] foreSE, backSE : foreground an background structuring elements
   * background structuring elements
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T fullThick(const Image<T> &imIn, const StrElt &foreSE,
                  const StrElt &backSE, Image<T> &imOut)
  {
    return fullThick(imIn, CompStrEltList(CompStrElt(foreSE, backSE)), imOut);
  }

  /** @} */

} // namespace smil

#endif // _D_THINNING_HPP
