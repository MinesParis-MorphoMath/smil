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
   * \ingroup Morpho
   * \defgroup Skeleton Skeleton
   * \{
   */

  /**
   * Skiz by influence zones
   *
   * Thinning of the background with a HMT_sL followed by a thinning with a
   * HMT_hM.
   */
  template <class T> RES_T skiz(const Image<T> &imIn, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freezer(imOut);
    Image<T> tmpIm(imIn);
    inv(imIn, imOut);
    fullThin(imOut, HMT_hL(6), tmpIm);
    fullThin(tmpIm, HMT_hM(6), imOut);

    return RES_OK;
  }

  /**
   * Morphological skeleton
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
   * Extinction values
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

    T2 r = 1;

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

  /**
   * Zhang 2D skeleton
   *
   * Implementation corresponding to the algorithm described in \cite
   * khanyile_comparative_2011.
   */
  template <class T> RES_T zhangSkeleton(const Image<T> &imIn, Image<T> &imOut)
  {
    size_t w = imIn.getWidth();
    size_t h = imIn.getHeight();

    // Create a copy image with a border to avoid border checks
    size_t width = w + 2, height = h + 2;
    Image<T> tmpIm(width, height);
    Image<T> modifiedIm(width, height);
    fill(tmpIm, ImDtTypes<T>::min());
    copy(imIn, tmpIm, 1, 1);
    copy(imIn, modifiedIm, 1, 1);
    write(tmpIm, "tmpIm.png");
    typedef typename Image<T>::sliceType sliceType;
    typedef typename Image<T>::lineType lineType;

    // lineType tab = tmpIm.getPixels();
    const sliceType lines         = tmpIm.getLines();
    const sliceType modifiedLines = modifiedIm.getLines();
    lineType curLine, modifiedLine;
    lineType curPix, modifiedPix;

    bool ptsDeleted1, ptsDeleted2, ptsDeleted;

    bool goOn;

    UINT nbrTrans, nbrNonZero;
    int iteration;

    int iWidth        = width;
    int ngbOffsets[8] = {-iWidth - 1, -iWidth, -iWidth + 1, 1,
                         iWidth + 1,  iWidth,  iWidth - 1,  -1};
    T ngbs[8];
    /// 0 1 2 OUR DEF
    /// 7    3
    /// 6 5  4
    // 2,4,6 (zhang) -> 1 3 5 (our) PHASE 1
    // 4,6,8 (zhang) -> 3 5 7 (our)

    // 2,4,8 (zhang) -> 1 3 7 (our) PHASE 2
    // 2,6,8 (zhang) -> 1 5 7 (our)

    /// 9 2 3 PAPER
    /// 8 1   4
    /// 7 6  5

    iteration = 0;
    int n;
    do {
      iteration   = iteration + 1;
      ptsDeleted1 = false;
      ptsDeleted2 = false;
      ptsDeleted  = false;

      // PHASE 1 south-east boundary and  north-west corner
      for (size_t y = 1; y < height - 1; y++) {
        curLine      = lines[y];
        curPix       = curLine + 1;
        modifiedLine = modifiedLines[y];
        modifiedPix  = modifiedLine + 1;
        for (size_t x = 1; x < width; x++, curPix++, modifiedPix++) {
          if (*curPix != 0) {
            for (n = 0; n < 8; n++) {
              ngbs[n] = *(curPix + ngbOffsets[n]);
            }

            goOn = false;

            // ----------------------------------------
            // Calculate the number of non-zero neighbors
            // ----------------------------------------
            nbrNonZero = 0;
            for (n = 0; n < 8; n++) {
              if (ngbs[n] != 0) {
                nbrNonZero++;
              }
            }
            if (nbrNonZero >= 2 && nbrNonZero <= 6)
              goOn = true;

            // --------------------------------------------------
            // Calculate the number of transitions in clockwise direction
            // from point (-1,-1) back to itself
            // --------------------------------------------------
            if (goOn) {
              nbrTrans = 0;
              for (n = 0; n < 7; n++)
                if (ngbs[n] == 0)
                  if (ngbs[n + 1] != 0)
                    nbrTrans++;
              if (ngbs[7] == 0)
                if (ngbs[0] != 0)
                  nbrTrans++;

              if (nbrTrans != 1)
                goOn = false;
            }

            // --------------------------------------------------
            // P1 , 3,5
            // --------------------------------------------------

            if (goOn) {
              if (ngbs[1] * ngbs[3] * ngbs[5] != 0) // phase 1
              {
                goOn = false;
              }
            }
            // --------------------------------------------------
            // P3 , 5,7
            // --------------------------------------------------
            if (goOn) {
              if (ngbs[3] * ngbs[5] * ngbs[7] != 0) // phase 1
              {
                goOn = false;
              }
            }
            // --------------------------------------------------
            // All conditions verified, remove the point
            // --------------------------------------------------
            if (goOn) {
              *modifiedPix = 0;
              //			    std::cout<<".......DELETE1 ("<<x<<","<<y<<")\n";
              ptsDeleted1 = true;
            }
          } // * curPix != 0
        }   // for x
      }     // for y

      // Delete pixels satisfying all previous conditions (phase 1)
      if (ptsDeleted1) {
        copy(modifiedIm, tmpIm);
      }

      // PHASE 2 north-west boundary  south-east corner
      for (size_t y = 1; y < height - 1; y++) {
        curLine      = lines[y];
        curPix       = curLine + 1;
        modifiedLine = modifiedLines[y];
        modifiedPix  = modifiedLine + 1;
        for (size_t x = 1; x < width; x++, curPix++, modifiedPix++) {
          if (*curPix != 0) {
            goOn = false;

            for (n = 0; n < 8; n++) {
              ngbs[n] = *(curPix + ngbOffsets[n]);
            }

            // --------------------------------------------------
            // Calculate the number of non-zero neighbors
            // --------------------------------------------------
            nbrNonZero = 0;
            for (n = 0; n < 8; n++)
              if (ngbs[n] != 0)
                nbrNonZero++;
            if (nbrNonZero >= 2 && nbrNonZero <= 6)
              goOn = true;

            // --------------------------------------------------
            // Calculate the number of transitions in clockwise direction
            // from point (-1,-1) back to itself
            // --------------------------------------------------
            if (goOn) {
              nbrTrans = 0;
              for (n = 0; n < 7; n++)
                if (ngbs[n] == 0)
                  if (ngbs[n + 1] != 0)
                    nbrTrans++;
              if (ngbs[7] == 0)
                if (ngbs[0] != 0)
                  nbrTrans++;
              if (nbrTrans != 1)
                goOn = false;
            }

            // --------------------------------------------------
            // P1 , 3,7
            // --------------------------------------------------
            if (goOn) {
              if (ngbs[1] * ngbs[3] * ngbs[7] != 0) // phase 2
              {
                goOn = false;
              }
            }
            // --------------------------------------------------
            // P1 , 5,7
            // --------------------------------------------------

            if (goOn) {
              if (ngbs[1] * ngbs[5] * ngbs[7] != 0) // phase 2
              {
                goOn = false;
              }
            }
            // --------------------------------------------------
            // All conditions verified, remove the point
            // --------------------------------------------------
            if (goOn) {
              *modifiedPix = 0;
              ptsDeleted2  = true;
              //			    std::cout<<".......DELETE2 ("<<x<<","<<y<<")\n";
            }
          } // * curPix != 0
        }   // for x
      }     // for y
      // Delete pixels satisfying all previous conditions (phase 2)
      if (ptsDeleted2) {
        copy(modifiedIm, tmpIm);
      }

      ptsDeleted = ptsDeleted1 || ptsDeleted2;

    } while (ptsDeleted); // do

    copy(tmpIm, 1, 1, imOut);

    return RES_OK;
  }

  /** \} */

} // namespace smil

#endif // _D_SKELETON_HPP
