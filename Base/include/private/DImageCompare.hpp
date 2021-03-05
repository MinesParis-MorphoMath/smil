/*
 * __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2021 Centre de Morphologie Mathematique
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
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   This file does... some very complex morphological operation...
 *
 * History :
 *   - 08/06/2020 - by Jose-Marcio Martins da Cruz
 *     Porting from xxx
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_IMAGE_COMPARE_HPP
#define _D_IMAGE_COMPARE_HPP

namespace smil
{
  /**
   * @ingroup Base
   * @defgroup DIndex Image Similarities
   *
   * @details This module provides functions to compare images similarities
   *
   *
   * @{ */

  /**
   * indexJaccard()
   *
   * - for binary images, this function returns the
   *    @TB{Jaccard similarity coefficient} defined by :
   * @f[
   *  Jaccard(imA, imB) = \dfrac{|imA \cap imB|}{\vert imA \cup imB \vert} =
   *    \dfrac{area(logicAnd(imA, \; imB))}{area(logicOr(imA, \; imB))}
   * @f]
   * - for non binary images, this function returns the
   *    @TB{Weighted Jaccard similarity coefficient}, also known as
   *    @TB{Ruzicka coefficient} - see indexRuzicka()
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Jaccard_index, Jaccard Index}
   *
   * @param[in] imIn1 : First input image
   * @param[in] imIn2 : Second input image
   * @returns Jaccard similarity coefficient between the two images
   */
  template <typename T>
  double indexJaccard(const Image<T> &imIn1, const Image<T> &imIn2)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2);
    ASSERT_SAME_SIZE(&imIn1, &imIn2);
    if (isBinary(imIn1) && isBinary(imIn2)) {
      Image<T> imOr(imIn1);
      Image<T> imAnd(imIn1);

      logicOr(imIn1, imIn2, imOr);
      logicAnd(imIn1, imIn2, imAnd);

      if (area(imOr) > 0)
        return double(area(imAnd)) / double(area(imOr));

      return 1.;
    }
    return 1.;
  }

  /**
   * indexRuzicka()
   *
   * Returns the @TB{Weighted Jaccard similarity coefficient}, also known as
   *    @TB{Ruzicka coefficient} - see indexJaccard()
   *
   * @f[
   *  Ruzicka(imA, imB) = \dfrac{\sum_{i,j,k} min(imA(i,j,k), \; imA(i,j,k))}
   *                            {\sum_{i,j,k} min(imA(i,j,k), \; imA(i,j,k))} =
   *                      \dfrac{volume(inf(imA, \; imB))}
   *                            {volume(sup(imA, \; imB))}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Jaccard_index, Jaccard Index}
   *
   * @param[in] imIn1 : First input image
   * @param[in] imIn2 : Second input image
   * @returns Ruzicka similarity coefficient between the two images
   */
  template <typename T>
  double indexRuzicka(const Image<T> &imIn1, const Image<T> &imIn2)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2);
    ASSERT_SAME_SIZE(&imIn1, &imIn2);

    Image<T> imMax(imIn1);
    Image<T> imMin(imIn1);

    sup(imIn1, imIn2, imMax);
    inf(imIn1, imIn2, imMin);

    if (volume(imMax) > 0)
      return double(volume(imMin)) / double(volume(imMax));

    return 1.;
  }

  /**
   * distanceHamming()
   *
   * Returns the number of pixels with different values in the two images.
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Hamming_distance, Hamming distance}
   *
   * @param[in] imIn1 : First input image
   * @param[in] imIn2 : Second input image
   * @returns Hamming distance between two images
   */
  template <typename T>
  size_t distanceHamming(const Image<T> &imIn1, const Image<T> &imIn2)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2);
    ASSERT_SAME_SIZE(&imIn1, &imIn2);

    Image<T> imOut(imIn1);

    diff(imIn1, imIn2, imOut);

    return area(imOut);
  }

  /** @} */

} // namespace smil

#endif // _D_IMAGE_COMPARE_HPP
