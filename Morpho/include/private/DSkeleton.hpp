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

#ifndef _D_SKELETON_HPP
#define _D_SKELETON_HPP

#include "DMorphoBase.hpp"
#include "DHitOrMiss.hpp"

namespace smil
{
  /**
   * @ingroup Morpho
   * @defgroup Skeleton Skeleton
   *
   * @details
   * @b Skeletons result from sequential iterations of thinnings with specific
   * composite SEs that generate a medial axis of the input set. This medial
   * axis are called skeleton. It consists of a compact representation which
   * preserves only those points of a set whose minimum distance to the
   * boundary of the set reaches at least two distinct boundary points.
   *
   * @see
   *  - @SoilleBook{Chap. 5}
   *
   * @{
   */

  /**
   * skiz() - Skeleton by Influence Zones @TB{(Skiz)}
   *
   * Thinning of the background with a Composite Structuring Element
   * @b HMT_sL(6) followed by a thinning with a Composite Structuring Element
   * @b HMT_hM(6).
   *
   * @see
   * - @SoilleBook{p. 170}
   * - @b HMT_sL() and @b HMT_hM() in CompStrEltList
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T skiz(const Image<T> &imIn, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freezer(imOut);
    Image<T>     tmpIm(imIn);
    inv(imIn, imOut);
    fullThin(imOut, HMT_hL(6), tmpIm);
    fullThin(tmpIm, HMT_hM(6), imOut);

    return RES_OK;
  }

  /**
   * skeleton() - Morphological skeleton
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T skeleton(const Image<T> &imIn, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freezer(imOut);

    Image<T> imEro(imIn);
    Image<T> imTemp(imIn);

    copy(imIn, imEro);
    fill(imOut, ImDtTypes<T>::min());

    bool idempt = false;

    do {
      erode(imEro, imEro, se);
      open(imEro, imTemp, se);
      sub(imEro, imTemp, imTemp);
      sup(imOut, imTemp, imTemp);
      idempt = equ(imTemp, imOut);
      copy(imTemp, imOut);
    } while (!idempt);

    return RES_OK;
  }

  /**
   * extinctionValues() - Extinction values
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T1, class T2>
  RES_T extinctionValues(const Image<T1> &imIn, Image<T2> &imOut,
                         const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freezer(imOut);

    Image<T1> imEro(imIn);
    Image<T1> imTemp1(imIn);
    Image<T2> imTemp2(imOut);

    copy(imIn, imEro);
    fill(imOut, ImDtTypes<T2>::min());

    T2   r      = 1;
    bool idempt = false;
    do {
      erode(imEro, imEro, se);
      open(imEro, imTemp1, se);
      sub(imEro, imTemp1, imTemp1);
      test(imTemp1, r++, imOut, imTemp2);
      idempt = equ(imTemp2, imOut);
      copy(imTemp2, imOut);
    } while (!idempt);

    return RES_OK;
  }


  /*
   *
   * #####   #####   #    #  #    #  ######
   * #    #  #    #  #    #  ##   #  #
   * #    #  #    #  #    #  # #  #  #####
   * #####   #####   #    #  #  # #  #
   * #       #   #   #    #  #   ##  #
   * #       #    #   ####   #    #  ######
   *
   */

  /**
   * pruneSkiz() -
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   *
   */
  template <class T>
  RES_T pruneSkiz(const Image<T> &imIn, Image<T> &imOut,
                  const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    T *in  = imIn.getPixels();
    T *out = imOut.getPixels();

    fill<T>(imOut, T(0));

    size_t Size[3];
    imIn.getSize(Size);

    size_t nbrPixels   = Size[0] * Size[1] * Size[2];
    size_t sePtsNumber = se.points.size();

#ifdef USE_OPEN_MP
    UINT nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef USE_OPEN_MP
#pragma omp for
#endif
      for (size_t i = 0; i < nbrPixels; ++i) {
        OffsetPoint pt(Size);
        pt.setOffset(i);

        bool up   = false;
        bool down = false;
        if (in[pt.o] > 0 && in[pt.o] != ImDtTypes<T>::max()) {
          for (UINT pts = 0; pts < sePtsNumber; ++pts) {
            OffsetPoint qt(Size);
            qt = pt;
            qt.shift(se.points[pts]);

            if (qt.inImage()) {
              if (in[qt.o] != ImDtTypes<T>::max()) {
                if (in[qt.o] >= in[pt.o] + 1) {
                  up = true;
                }
                if (in[qt.o] <= in[pt.o] - 1) {
                  down = true;
                }
              }
            }
          }

          if (!up || !down) {
            out[pt.o] = in[pt.o];
          }
        }
      }
    }

    return RES_OK;
  }

  /** @} */

} // namespace smil

#endif // _D_SKELETON_HPP
