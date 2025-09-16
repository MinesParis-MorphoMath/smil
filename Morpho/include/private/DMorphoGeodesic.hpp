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

#ifndef _D_MORPHO_GEODESIC_HPP
#define _D_MORPHO_GEODESIC_HPP

#include "DMorphImageOperations.hpp"
#include "DMorphoHierarQ.hpp"
#include "Base/include/private/DImageDraw.hpp"
#include "Base/include/private/DImageHistogram.hpp"
#include "Morpho/include/private/DMorphoBase.hpp"

namespace smil
{
  /**
   * @ingroup Morpho
   * @defgroup Geodesic Geodesic Transforms
   *
   * Geodesic Transformations
   *
   * @see
   * - <a href=https://nbviewer.jupyter.org/url/smil.cmm.mines-paristech.fr/notebooks/cours/Geodesy.ipynb>Geodesy - Morphological Mathematics courses at Mines-Paristech</a>
   * - @SoilleBook{Chap. 6}
   *
   * @{
   */

  // Geodesy
  /*
   * ######  #    #  #    #   ####    #####     #     ####   #    #   ####
   * #       #    #  ##   #  #    #     #       #    #    #  ##   #  #
   * #####   #    #  # #  #  #          #       #    #    #  # #  #   ####
   * #       #    #  #  # #  #          #       #    #    #  #  # #       #
   * #       #    #  #   ##  #    #     #       #    #    #  #   ##  #    #
   * #        ####   #    #   ####      #       #     ####   #    #   ####
   */
  /**
   * geoDilate() - Geodesic dilation
   *
   * @param[in] imIn : input image
   * @param[in] imMask : mask
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T geoDilate(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut,
                  const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

    ImageFreezer freeze(imOut);

    ASSERT((inf(imIn, imMask, imOut) == RES_OK));
    StrElt tmpSE(se(1));

    for (UINT i = 0; i < se.size; i++) {
      ASSERT((dilate<T>(imOut, imOut, tmpSE) == RES_OK));
      ASSERT((inf(imOut, imMask, imOut) == RES_OK));
    }
    return RES_OK;
  }

  /** @cond */
  template <class T>
  RES_T geoDil(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut,
               const StrElt &se = DEFAULT_SE)
  {
    return geoDilate(imIn, imMask, imOut, se);
  }
  /** @endcond */

  /**
   * geoErode() - Geodesic erosion
   *
   * @param[in] imIn : input image
   * @param[in] imMask : mask
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T geoErode(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

    ImageFreezer freeze(imOut);

    ASSERT((sup(imIn, imMask, imOut) == RES_OK));
    StrElt tmpSE(se(1));

    for (UINT i = 0; i < se.size; i++) {
      ASSERT((erode(imOut, imOut, tmpSE) == RES_OK));
      ASSERT((sup(imOut, imMask, imOut) == RES_OK));
    }
    return RES_OK;
  }

  /** @cond */
  template <class T>
  RES_T geoEro(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut,
               const StrElt &se = DEFAULT_SE)
  {
    return geoErode(imIn, imMask, imOut, se);
  }
  /** @endcond */

  /**
   * geoBuild() - Geodesic Reconstruction
   *
   * @param[in] imIn : input image
   * @param[in] imMask : mask
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T geoBuild(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

    ImageFreezer freeze(imOut);

    ASSERT((inf(imIn, imMask, imOut) == RES_OK));

    double vol1 = vol(imOut), vol2;
    while (true) {
      ASSERT((dilate<T>(imOut, imOut, se) == RES_OK));
      ASSERT((inf(imOut, imMask, imOut) == RES_OK));
      vol2 = vol(imOut);
      if (vol2 == vol1)
        break;
      vol1 = vol2;
    }
    return RES_OK;
  }

  /**
   * geoDualBuild() - Geodesic Dual Reconstruction
   *
   * @param[in] imIn : input image
   * @param[in] imMask : mask
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T geoDualBuild(const Image<T> &imIn, const Image<T> &imMask,
                     Image<T> &imOut, const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

    ImageFreezer freeze(imOut);

    ASSERT((sup(imIn, imMask, imOut) == RES_OK));

    double vol1 = vol(imOut), vol2;

    while (true) {
      ASSERT((erode(imOut, imOut, se) == RES_OK));
      ASSERT((sup(imOut, imMask, imOut) == RES_OK));
      vol2 = vol(imOut);
      if (vol2 == vol1)
        break;
      vol1 = vol2;
    }
    return RES_OK;
  }

  /*
   * #    #   #   ######  #####          ####   #    #  ######  #    #  ######
   * #    #   #   #       #    #        #    #  #    #  #       #    #  #
   * ######   #   #####   #    #  ###   #    #  #    #  #####   #    #  ####
   * #    #   #   #       #####         #  # #  #    #  #       #    #  #
   * #    #   #   #       #   #         #   #   #    #  #       #    #  #
   * #    #   #   ######  #    #         ### #   ####   ######   ####   ######
   */
  /** @cond */
  template <class T>
  RES_T initBuildHierarchicalQueue(const Image<T> &imIn,
                                   HierarchicalQueue<T> &hq)
  {
    // Initialize the priority queue
    hq.initialize(imIn);

    typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();

    size_t s[3];

    imIn.getSize(s);
    size_t offset = 0;

    for (size_t i = 0; i < imIn.getPixelCount(); i++) {
      hq.push(*inPixels, offset);
      inPixels++;
      offset++;
    }

    //     hq.printSelf();
    return RES_OK;
  }

  template <class T>
  RES_T initBuildHierarchicalQueue(const Image<T> &imIn,
                                   HierarchicalQueue<T> &hq,
                                   const T noPushValue)
  {
    // Initialize the priority queue
    hq.initialize(imIn);

    typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();

    size_t s[3];

    imIn.getSize(s);
    size_t offset = 0;

    for (size_t i = 0; i < imIn.getPixelCount(); i++) {
      if (*inPixels != noPushValue) {
        hq.push(*inPixels, offset);
      }
      inPixels++;
      offset++;
    }

    //     hq.printSelf();
    return RES_OK;
  }

  template <class T, class operatorT>
  RES_T processBuildHierarchicalQueue(Image<T> &imIn, const Image<T> &imMask,
                                      Image<UINT8> &imStatus,
                                      HierarchicalQueue<T> &hq,
                                      const StrElt &se)
  {
    typename ImDtTypes<T>::lineType inPixels       = imIn.getPixels();
    typename ImDtTypes<T>::lineType markPixels     = imMask.getPixels();
    typename ImDtTypes<UINT8>::lineType statPixels = imStatus.getPixels();

    std::vector<int> dOffsets;
    operatorT oper;

    std::vector<IntPoint>::const_iterator it_start = se.points.begin();
    std::vector<IntPoint>::const_iterator it_end   = se.points.end();
    std::vector<IntPoint>::const_iterator it;

    std::vector<size_t> tmpOffsets;

    size_t s[3];
    imIn.getSize(s);

    // set an offset distance for each se point
    for (it = it_start; it != it_end; it++) {
      dOffsets.push_back(it->x + it->y * s[0] + it->z * s[0] * s[1]);
    }

    std::vector<int>::iterator it_off_start = dOffsets.begin();
    std::vector<int>::iterator it_off;

    size_t x0, y0, z0;
    size_t curOffset;

    int x, y, z;
    size_t nbOffset;
    UINT8 nbStat;

    while (!hq.isEmpty()) {
      curOffset = hq.pop();

      // Give the point the label "FINAL" in the status image
      statPixels[curOffset] = HQ_FINAL;

      imIn.getCoordsFromOffset(curOffset, x0, y0, z0);

      bool oddLine = se.odd && (y0) % 2;

      for (it = it_start, it_off = it_off_start; it != it_end; it++, it_off++)
        if (it->x != 0 || it->y != 0 ||
            it->z != 0) // useless if x=0 & y=0 & z=0
        {
          x = x0 + it->x;
          y = y0 + it->y;
          z = z0 + it->z;

          if (oddLine)
            x += ((y + 1) % 2 != 0);

          if (x >= 0 && x < (int) s[0] && y >= 0 && y < (int) s[1] && z >= 0 &&
              z < (int) s[2]) {
            nbOffset = curOffset + *it_off;

            if (oddLine)
              nbOffset += ((y + 1) % 2 != 0);

            nbStat = statPixels[nbOffset];

            if (nbStat == HQ_CANDIDATE) {
              inPixels[nbOffset] =
                  oper(inPixels[curOffset], markPixels[nbOffset]);
              statPixels[nbOffset] = HQ_QUEUED;
              hq.push(inPixels[nbOffset], nbOffset);
            }
          }
        }
    }
    return RES_OK;
  }

  template <class T> struct minFunctor {
    inline T operator()(T a, T b)
    {
      return std::min(a, b);
    }
  };

  template <class T> struct maxFunctor {
    inline T operator()(T a, T b)
    {
      return std::max(a, b);
    }
  };
  /** @endcond */

  /**
   * dualBuild() - Reconstruction by erosion - dual build - (using hierarchical queues).
   *
   * @param[in] imIn : input image
   * @param[in] imMask : mask
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T dualBuild(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut,
                  const StrElt &se = DEFAULT_SE)
  {
    //         if (isBinary(imIn) && isBinary(imMask))
    //           return dualBinBuild(imIn, imMask, imOut, se);

    ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

    ImageFreezer freeze(imOut);

    Image<UINT8> imStatus(imIn);
    HierarchicalQueue<T> pq;

    // Make sure that imIn >= imMask
    ASSERT((sup(imIn, imMask, imOut) == RES_OK));

    // Set all pixels in the status image to CANDIDATE
    ASSERT((fill(imStatus, (UINT8) HQ_CANDIDATE) == RES_OK));

    // Initialize the PQ
    initBuildHierarchicalQueue(imOut, pq);
    processBuildHierarchicalQueue<T, maxFunctor<T>>(imOut, imMask, imStatus, pq,
                                                    se);

    return RES_OK;
  }

  /**
   * build() - Reconstruction by dilation (using hierarchical queues).
   *
   * @param[in] imIn : input image
   * @param[in] imMask : mask
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T build(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut,
              const StrElt &se = DEFAULT_SE)
  {
    if (isBinary(imIn) && isBinary(imMask))
      return binBuild(imIn, imMask, imOut, se);

    ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

    ImageFreezer freeze(imOut);

    Image<UINT8> imStatus(imIn);

    // Reverse hierarchical queue (the highest token corresponds to the highest
    // gray value)
    HierarchicalQueue<T> rpq(true);

    // Make sure that imIn <= imMask
    ASSERT((inf(imIn, imMask, imOut) == RES_OK));

    // Set all pixels in the status image to CANDIDATE
    ASSERT((fill(imStatus, (UINT8) HQ_CANDIDATE) == RES_OK));

    // Initialize the PQ
    initBuildHierarchicalQueue(imOut, rpq);
    processBuildHierarchicalQueue<T, minFunctor<T>>(imOut, imMask, imStatus,
                                                    rpq, se);

    return RES_OK;
  }

  /**
   * binBuild() - Reconstruction (using hierarchical queues).
   *
   * @param[in] imIn : input image
   * @param[in] imMask : mask
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T binBuild(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);
    //        T noPushValue = NUMERIC_LIMITS<T>::min();
    // T maxValue =  NUMERIC_LIMITS<T>::max();
    ImageFreezer freeze(imOut);

    Image<UINT8> imStatus(imIn);

    // Reverse hierarchical queue (the highest token corresponds to the highest
    // gray value)
    HierarchicalQueue<T> rpq(true);

    // Make sure that imIn <= imMask
    ASSERT((inf(imIn, imMask, imOut) == RES_OK));

    // make a status image with all foreground pixels as CANDIDATE, otherwise as
    // FINAL
    ASSERT(test(imMask, (UINT8) HQ_CANDIDATE, (UINT8) HQ_FINAL, imStatus) ==
           RES_OK);

    // Initialize the PQ
    initBuildHierarchicalQueue(imOut, rpq, imOut.getDataTypeMin());
    processBuildHierarchicalQueue<T, minFunctor<T>>(imOut, imMask, imStatus,
                                                    rpq, se);

    return RES_OK;
  }

  //     /**
  //     * Reconstruction (using hierarchical queues).
  //     */
  //     template <class T>
  //     RES_T dualBinBuild(const Image<T> &imIn, const Image<T> &imMask,
  //     Image<T> &imOut, const StrElt &se=DEFAULT_SE)
  //     {
  //         ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
  //         ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);
  //         //        T noPushValue = NUMERIC_LIMITS<T>::min();
  //         //T maxValue =  NUMERIC_LIMITS<T>::max();
  //         ImageFreezer freeze(imOut);
  //
  //         Image<UINT8> imStatus(imIn);
  //
  //         HierarchicalQueue<T> rpq;
  //
  //         // Make sure that imIn >= imMask
  //         ASSERT((sup(imIn, imMask, imOut)==RES_OK));
  //
  //         // make a status image with all background pixels as CANDIDATE,
  //         otherwise as FINAL ASSERT(test(imMask, (UINT8)HQ_FINAL,
  //         (UINT8)HQ_CANDIDATE, imStatus)==RES_OK);
  //
  //         // Initialize the PQ
  //         initBuildHierarchicalQueue(imOut, rpq, imOut.getDataTypeMin());
  //         processBuildHierarchicalQueue<T, maxFunctor<T> >(imOut, imMask,
  //         imStatus, rpq, se);
  //
  //         return RES_OK;
  //     }

  /**
   * hBuild() - h-Reconstuction
   *
   * Performs a subtraction of size @b height followed by a reconstruction
   *
   * @param[in] imIn : input image
   * @param[in] height : value to be subtracted to the image values
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T hBuild(const Image<T> &imIn, const T &height, Image<T> &imOut,
               const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    if (&imIn == &imOut) {
      Image<T> tmpIm = imIn;
      return hBuild(tmpIm, height, imOut, se);
    }

    ImageFreezer freeze(imOut);

    ASSERT((sub(imIn, T(height), imOut) == RES_OK));
    ASSERT((build(imOut, imIn, imOut, se) == RES_OK));

    return RES_OK;
  }

  /**
   * hDualBuild() - Dual h-Reconstuction
   *
   * Performs an addition of size @b height followed by a dual reconstruction
   *
   * @param[in] imIn : input image
   * @param[in] height : value to be added to the image values
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T hDualBuild(const Image<T> &imIn, const T &height, Image<T> &imOut,
                   const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    if (&imIn == &imOut) {
      Image<T> tmpIm = imIn;
      return hDualBuild(tmpIm, height, imOut, se);
    }

    ImageFreezer freeze(imOut);

    ASSERT((add(imIn, T(height), imOut) == RES_OK));
    ASSERT((dualBuild(imOut, imIn, imOut, se) == RES_OK));

    return RES_OK;
  }

  /**
   * buildOpen() - Opening by reconstruction
   *
   * Erosion followed by a reconstruction (build)
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T buildOpen(const Image<T> &imIn, Image<T> &imOut,
                  const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn);
    ASSERT((erode(imIn, tmpIm, se) == RES_OK));
    ASSERT((build(tmpIm, imIn, imOut, se) == RES_OK));

    return RES_OK;
  }

  /**
   * buildClose() - Closing by reconstruction
   *
   * Dilation followed by a reconstruction (dualBuild)
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T buildClose(const Image<T> &imIn, Image<T> &imOut,
                   const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn);
    ASSERT((dilate(imIn, tmpIm, se) == RES_OK));
    ASSERT((dualBuild(tmpIm, imIn, imOut, se) == RES_OK));

    return RES_OK;
  }

  /**
   * Alternate Sequential reconstructions beginning by a buildOpen
   *
   * Sequence of buildOpen() and buildClose() with increasing size <b>1, 2, ...,
   * max_size</b>. The @b max_size is given by the size of the structuring
   * element (for example @b 3 for @b SE(3)).
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element with the maximum size of the filter
   *
   */
  template <class T>
  RES_T asBuildOpen(const Image<T> &imIn, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn, true); // clone
    for (UINT i = 1; i <= se.size; i++) {
      ASSERT((buildOpen(tmpIm, imOut, se(i)) == RES_OK));
      ASSERT((buildClose(imOut, tmpIm, se(i)) == RES_OK));
    }
    ASSERT((copy(tmpIm, imOut) == RES_OK));

    return RES_OK;
  }


  /**
   * Alternate Sequential reconstructions beginning by a buildClose
   *
   * Sequence of buildClose() and buildOpen() with increasing size <b>1, 2, ...,
   * max_size</b>. The @b max_size is given by the size of the structuring
   * element (for example @b 3 for @b SE(3)).
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element with the maximum size of the filter
   *
   * @smilexample{example-asfclose.py}
   */
  template <class T>
  RES_T asBuildClose(const Image<T> &imIn, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn, true); // clone
    for (UINT i = 1; i <= se.size; i++) {
      ASSERT((buildClose(tmpIm, imOut, se(i)) == RES_OK));
      ASSERT((buildOpen(imOut, tmpIm, se(i)) == RES_OK));
    }
    ASSERT((copy(tmpIm, imOut) == RES_OK));

    return RES_OK;
  }

  /**
   * fillHoles() - Hole filling
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T fillHoles(const Image<T> &imIn, Image<T> &imOut,
                  const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn);

    ASSERT((fill(tmpIm, ImDtTypes<T>::max()) == RES_OK));

#if 1
    off_t w = tmpIm.getWidth();
    off_t h = tmpIm.getHeight();
    off_t d = tmpIm.getDepth();

    T bVal = ImDtTypes<T>::min();
    if (d > 2) {
      for (off_t j = 0; j < h; j++) {
        for (off_t i = 0; i < w; i++) {
          tmpIm.setPixel(i, j, 0, bVal);
          tmpIm.setPixel(i, j, d - 1, bVal);
        }
      }
    }
    if (h > 2) {
      for (off_t k = 0; k < d; k++) {
        for (off_t i = 0; i < w; i++) {
          tmpIm.setPixel(i, 0, k, bVal);
          tmpIm.setPixel(i, h - 1, k, bVal);
        }
      }
    }
    if (w > 2) {
      for (off_t k = 0; k < d; k++) {
        for (off_t j = 0; j < h; j++) {
          tmpIm.setPixel(0, j, k, bVal);
          tmpIm.setPixel(w - 1, j, k, bVal);
        }
      }
    }
#else
    ASSERT((drawRectangle(tmpIm, 0, 0, tmpIm.getWidth(), tmpIm.getHeight(),
                          ImDtTypes<T>::min()) == RES_OK));
#endif
    ASSERT((dualBuild(tmpIm, imIn, imOut, se) == RES_OK));

    return RES_OK;
  }

  /**
   * levelPics() - Dual hole filling
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T levelPics(const Image<T> &imIn, Image<T> &imOut,
                  const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn);
    ASSERT((inv(imIn, tmpIm) == RES_OK));
    ASSERT((fillHoles(tmpIm, imOut, se) == RES_OK));
    ASSERT((inv(imOut, imOut) == RES_OK));

    //     return res;
    return RES_OK;
  }

  /** @} */

} // namespace smil

#endif // _D_MORPHO_GEODESIC_HPP
