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

#ifndef _D_ZHANG_SKEL_HPP
#define _D_ZHANG_SKEL_HPP

//#include "DCore.h"
#include "Core/include/DCore.h"

//#include "DMorphoBase.hpp"
//#include "DHitOrMiss.hpp"

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup AddonZhangSkel
   *
   * @details
   * @b Skeletons result from sequential iterations of thinnings that generate
   * a medial axis of the input set. This medial axis are called skeleton.
   *
   * Algorithms in this Addon don't make use of mathematical morphology
   * operations but, most of the time, on empirical rules to iteratively
   * removal of pixels based on their @TI{connectivity} or
   * @TI{neighborhood} till idempotency.
   *
   * @see
   *  - @SoilleBook{Chap. 5}
   *  - @TB{Some related papers} :
   *    @cite blum1967transformation,
   *    @cite zhang_suen_1984,
   *    @cite Chen_Hsu_1988_99,
   *    @cite Huang_Wan_Liu_2003,
   *    @cite khanyile_comparative_2011,
   *    @cite dong_lin_huang_2016 or
   *    @cite jain2017sequential
   *  - For creating skeletons based on mathematical morphology operators, see
   * skeleton(), skiz()
   *
   * @{
   */

#if 0
  /** @cond */
  /**
   * zhangSkeleton() - Zhang @b 2D skeleton
   *
   * Implementation corresponding to the algorithm described in
   * @cite zhang_suen_1984, @cite Chen_HSU_1988_99 and
   * @cite khanyile_comparative_2011.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   * @note
   * - @b 2D only
   */
  template <class T>
  RES_T zhangSkeleton(const Image<T> &imIn, Image<T> &imOut)
  {
    size_t w = imIn.getWidth();
    size_t h = imIn.getHeight();

    // Create a copy image with a border to avoid border checks
    size_t   width = w + 2, height = h + 2;
    Image<T> tmpIm(width, height);
    Image<T> modifiedIm(width, height);

    fill(tmpIm, ImDtTypes<T>::min());
    fill(modifiedIm, ImDtTypes<T>::min());

    copy(imIn, tmpIm, 1, 1);
    copy(imIn, modifiedIm, 1, 1);
    // write(tmpIm, "tmpIm.png");

    typedef typename Image<T>::sliceType sliceType;
    typedef typename Image<T>::lineType  lineType;

    // lineType tab = tmpIm.getPixels();
    const sliceType lines         = tmpIm.getLines();
    const sliceType modifiedLines = modifiedIm.getLines();
    lineType        curLine, modifiedLine;
    lineType        curPix, modifiedPix;

    bool ptsDeleted1, ptsDeleted2, ptsDeleted;

    bool goOn;

    UINT nbrTrans, nbrNonZero;
    int  iteration;

    int iWidth        = width;
    int ngbOffsets[8] = {-iWidth - 1, -iWidth, -iWidth + 1, 1,
                         iWidth + 1,  iWidth,  iWidth - 1,  -1};
    T   ngbs[8];

    /*
     * 0  1  2 OUR DEF
     * 7     3
     * 6  5  4
     *
     * 2  4  6 (zhang) -> 1 3 5 (our) PHASE 1
     * 4  6  8 (zhang) -> 3 5 7 (our)
     *
     * 2  4  8 (zhang) -> 1 3 7 (our) PHASE 2
     * 2  6  8 (zhang) -> 1 5 7 (our)
     *
     * 9  2  3 PAPER
     * 8  1  4
     * 7  6  5
     */

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

            if (!goOn)
              continue;

            // --------------------------------------------------
            // Calculate the number of transitions in clockwise direction
            // from point (-1,-1) back to itself
            // --------------------------------------------------
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

            if (!goOn)
              continue;

            // --------------------------------------------------
            // P1 , 3,5
            // --------------------------------------------------

            if (ngbs[1] * ngbs[3] * ngbs[5] != 0) // phase 1
              goOn = false;

            if (!goOn)
              continue;

            // --------------------------------------------------
            // P3 , 5,7
            // --------------------------------------------------
            if (ngbs[3] * ngbs[5] * ngbs[7] != 0) // phase 1
              goOn = false;

            if (!goOn)
              continue;

            // --------------------------------------------------
            // All conditions verified, remove the point
            // --------------------------------------------------
            *modifiedPix = 0;
            //  std::cout << ".......DELETE1 (" << x << "," << y << ")\n";
            ptsDeleted1 = true;

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
            for (n = 0; n < 8; n++) {
              if (ngbs[n] != 0)
                nbrNonZero++;
            }
            if (nbrNonZero >= 2 && nbrNonZero <= 6)
              goOn = true;

            if (!goOn)
              continue;

            // --------------------------------------------------
            // Calculate the number of transitions in clockwise direction
            // from point (-1,-1) back to itself
            // --------------------------------------------------
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

            if (!goOn)
              continue;

            // --------------------------------------------------
            // P1 , 3,7
            // --------------------------------------------------
            if (ngbs[1] * ngbs[3] * ngbs[7] != 0) // phase 2
              goOn = false;

            if (!goOn)
              continue;

            // --------------------------------------------------
            // P1 , 5,7
            // --------------------------------------------------

            if (ngbs[1] * ngbs[5] * ngbs[7] != 0) // phase 2
              goOn = false;

            if (!goOn)
              continue;

            // --------------------------------------------------
            // All conditions verified, remove the point
            // --------------------------------------------------
            *modifiedPix = 0;
            ptsDeleted2  = true;
            // std::cout << ".......DELETE2 (" << x << "," << y << ")\n";
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
  /** @endcond */
#endif

  /**
   * zhangSkeleton() - Zhang @b 2D skeleton
   *
   * Implementation corresponding to the algorithm described in
   * @cite zhang_suen_1984, @cite Chen_HSU_1988_99 and
   * @cite khanyile_comparative_2011.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   * @note
   * - @b 2D only
   */
  template <class T>
  RES_T zhangSkeleton(const Image<T> &imIn, Image<T> &imOut)
  {
    size_t w = imIn.getWidth();
    size_t h = imIn.getHeight();

    // Create a copy image with a border to avoid border checks
    size_t   width = w + 2, height = h + 2;
    Image<T> tmpIm(width, height);
    Image<T> modifiedIm(width, height);

    fill(tmpIm, ImDtTypes<T>::min());
    fill(modifiedIm, ImDtTypes<T>::min());

    copy(imIn, tmpIm, 1, 1);
    copy(imIn, modifiedIm, 1, 1);

    typedef typename Image<T>::sliceType sliceType;
    typedef typename Image<T>::lineType  lineType;

    const sliceType lines         = tmpIm.getLines();
    const sliceType modifiedLines = modifiedIm.getLines();
    lineType        curLine, modifiedLine;
    lineType        curPix, modifiedPix;

    bool ptsDeleted1, ptsDeleted2;

    UINT nbrTrans, nbrNonZero;
    int  iteration;

    int iWidth        = width;
    int ngbOffsets[8] = {-iWidth - 1, -iWidth, -iWidth + 1, 1,
                         iWidth + 1,  iWidth,  iWidth - 1,  -1};
    T   ngbs[8];

    // 0  1  2 OUR DEF  ->  9  2  3 PAPER
    // 7     3          ->  8  1  4
    // 6  5  4          ->  7  6  5

    // PHASE 1
    // 2  4  6 (zhang) -> 1 3 5 (our)
    // 4  6  8 (zhang) -> 3 5 7 (our)

    // PHASE 2
    // 2  4  8 (zhang) -> 1 3 7 (our) PHASE 2
    // 2  6  8 (zhang) -> 1 5 7 (our)

    iteration = 0;

    int n;
    do {
      iteration += 1;
      ptsDeleted1 = false;
      ptsDeleted2 = false;

      // PHASE 1 south-east boundary and  north-west corner
      for (size_t y = 1; y < height - 1; y++) {
        curLine      = lines[y];
        curPix       = curLine + 1;
        modifiedLine = modifiedLines[y];
        modifiedPix  = modifiedLine + 1;

        for (size_t x = 1; x < width; x++, curPix++, modifiedPix++) {
          if (*curPix == 0)
            continue;

          for (n = 0; n < 8; n++) {
            ngbs[n] = curPix[ngbOffsets[n]];
          }

          // ----------------------------------------
          // Calculate the number of non-zero neighbors
          // ----------------------------------------
          nbrNonZero = 0;
          for (n = 0; n < 8; n++) {
            if (ngbs[n] != 0) {
              nbrNonZero++;
            }
          }
          if (nbrNonZero < 2 || nbrNonZero > 6)
            continue;

          // --------------------------------------------------
          // Calculate the number of transitions in clockwise
          // direction from point (-1,-1) back to itself
          // --------------------------------------------------
          nbrTrans = 0;
          for (n = 0; n < 7; n++)
            if (ngbs[n] == 0 && ngbs[n + 1] != 0)
              nbrTrans++;
          if (ngbs[7] == 0 && ngbs[0] != 0)
            nbrTrans++;

          if (nbrTrans != 1)
            continue;

          // --------------------------------------------------
          // P 1, 3, 5
          // --------------------------------------------------
          if (ngbs[1] * ngbs[3] * ngbs[5] != 0) // phase 1
            continue;

          // --------------------------------------------------
          // P 3, 5, 7
          // --------------------------------------------------
          if (ngbs[3] * ngbs[5] * ngbs[7] != 0) // phase 1
            continue;

          // --------------------------------------------------
          // All conditions verified, remove the point
          // --------------------------------------------------
          *modifiedPix = 0;
          ptsDeleted1  = true;

        } // for x
      }   // for y

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
          if (*curPix == 0)
            continue;

          for (n = 0; n < 8; n++) {
            ngbs[n] = curPix[ngbOffsets[n]];
          }

          // --------------------------------------------------
          // Calculate the number of non-zero neighbors
          // --------------------------------------------------
          nbrNonZero = 0;
          for (n = 0; n < 8; n++) {
            if (ngbs[n] != 0)
              nbrNonZero++;
          }
          if (nbrNonZero < 2 || nbrNonZero > 6)
            continue;

          // --------------------------------------------------
          // Calculate the number of transitions in clockwise
          // direction from point (-1,-1) back to itself
          // --------------------------------------------------
          nbrTrans = 0;
          for (n = 0; n < 7; n++)
            if (ngbs[n] == 0 && ngbs[n + 1] != 0)
              nbrTrans++;
          if (ngbs[7] == 0 && ngbs[0] != 0)
            nbrTrans++;
          if (nbrTrans != 1)
            continue;

          // --------------------------------------------------
          // P 1, 3, 7
          // --------------------------------------------------
          if (ngbs[1] * ngbs[3] * ngbs[7] != 0) // phase 2
            continue;

          // --------------------------------------------------
          // P 1, 5, 7
          // --------------------------------------------------
          if (ngbs[1] * ngbs[5] * ngbs[7] != 0) // phase 2
            continue;

          // --------------------------------------------------
          // All conditions verified, remove the point
          // --------------------------------------------------
          *modifiedPix = 0;
          ptsDeleted2  = true;
        } // for x
      }   // for y

      // Delete pixels satisfying all previous conditions (phase 2)
      if (ptsDeleted2) {
        copy(modifiedIm, tmpIm);
      }
    } while (ptsDeleted1 || ptsDeleted2); // do

    copy(tmpIm, 1, 1, imOut);

    return RES_OK;
  }

  /**
   * zhangThinning() - Zhang @b 2D skeleton
   *
   * Implementation corresponding to the algorithm described in
   * @cite zhang_suen_1984, @cite Chen_HSU_1988_99 and
   * @cite khanyile_comparative_2011.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   * @note
   * - @b 2D only
   * @see zhangSkeleton()
   * @overload
   */
  template <class T>
  RES_T zhangThinning(const Image<T> &imIn, Image<T> &imOut)
  {
    return zhangSkeleton(imIn, imOut);
  }

  /** @cond */

  template <typename T>
  class ZhangThinning
  {
  public:
    ZhangThinning()
    {
      _init();
    }

  private:
    vector<IntPoint> nbOff{8};

    T minV;
    T maxV;

    off_t width;
    off_t height;
    off_t depth;

    void _init()
    {
      nbOff.resize(8);
      /*
       * Neighbor numbering
       *
       * OUR DEF      ZHANG PAPER
       * 0  1  2  ->  9  2  3
       * 7     3  ->  8  1  4
       * 6  5  4  ->  7  6  5
       *
       */
      nbOff[0] = {-1, -1, 0};
      nbOff[1] = {0, -1, 0};
      nbOff[2] = {1, -1, 0};
      nbOff[3] = {1, 0, 0};
      nbOff[4] = {1, 1, 0};
      nbOff[5] = {0, 1, 0};
      nbOff[6] = {-1, 1, 0};
      nbOff[7] = {-1, 0, 0};
    }

    void _init(const Image<T> &im)
    {
      _init();

      width  = im.getWidth();
      height = im.getHeight();
      depth  = im.getDepth();

      minV = minVal(im);
      maxV = maxVal(im);
    }

    T getNeighborValue(T *buf, off_t offset, off_t x, off_t y, IntPoint &nbg)
    {
      if (nbg.x < 0 && x == 0)
        return 0;
      if (nbg.x > 0 && x == width - 1)
        return 0;
      if (nbg.y < 0 && y == 0)
        return 0;
      if (nbg.y > 0 && y == height - 1)
        return 0;
      return buf[offset + nbg.x + nbg.y * width];
    }

    void getNeighborhood(Image<T> &im, off_t x, off_t y,
                                T nghbs[], int &Xr, int &Bp)
    {
      Xr = 0;
      Bp = 0;
      T prev;

      T *   buf       = im.getPixels();
      off_t pixOffset = x + y * width;

      for (auto i = 0; i < 8; i++) {
        nghbs[i] = getNeighborValue(buf, pixOffset, x, y, nbOff[i]);
        if (nghbs[i] != minV)
          Bp++;

        if (i == 0)
          prev = nghbs[i];
        if (prev != nghbs[i])
          Xr++;
        prev = nghbs[i];
      }
      // Xr is always pair
      Xr += (Xr % 2);
    }

  public:
    /*
     * AN IMPROVED PARALLEL THINNING ALGORITHM
     * JIANWEI DONG, WUHONG LIN, CHAO HUANG
     */
    RES_T skDongLinHuang(const Image<T> &imIn, Image<T> &imOut)
    {
      _init(imIn);

      if (depth > 1) {
        ERR_MSG("Only 2D images");
        return RES_ERR;
      }
      if (!isBinary(imIn)) {
        ERR_MSG("Only binary images");
        return RES_ERR;
      }

      Image<T> imTmp(imIn, true);
      Image<T> imMod(imIn);

      /* P1 to P8 - in this order
       *
       * P4 P3 P2      P0 P1 P2
       * P5    P1  =>  P7    P3
       * P6 P7 P8      P6 P5 P4
       *
       * OBS - Index begins with "1" in the paper
       */

      T *bufTmp = imTmp.getPixels();
      T *bufMod = imMod.getPixels();

      /* P R E - T H I N N I N G */
      copy(imTmp, imMod);
      for (off_t y = 0; y < height; y++) {
        for (off_t x = 0; x < width; x++) {
          int Bodd = 0;
          for (size_t i = 0; i < nbOff.size(); i += 2) {
            if (!imTmp.areCoordsInImage(x + nbOff[i].x, y + nbOff[i].y, 0))
              continue;
            if (bufTmp[(y + nbOff[i].y) * width + x + nbOff[x].x] != 0)
              Bodd++;
          }
          continue;
          if (Bodd < 2) {
            bufMod[y * width + x] = minV;
            continue;
          }
          if (Bodd > 2) {
            bufMod[y * width + x] = maxV;
            continue;
          }
        }
      }
      copy(imMod, imTmp);

      /* M A I N     L O O P */
      int  iteration = 0;
      bool done      = true;
      bool modified;
      do {
        iteration++;

        if (iteration > 2000) {
          cout << ">>> Iteraction limit reached : " << iteration << endl;
          break;
        }
        int Xr = 0;
        int Bp = 0;
        done   = true;

        // first sub-iteration
        modified = false;
        for (off_t y = 0; y < height; y++) {
          off_t lineOffset = y * width;

#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
          for (off_t x = 0; x < width; x++) {
            off_t     pixOffset = lineOffset + x;

            if (bufTmp[pixOffset] == minV)
              continue;

            T nghbs[8];
            getNeighborhood(imTmp, x, y, nghbs, Xr, Bp);
            if (Bp < 2 || Bp > 6)
              continue;
            if (Xr != 2)
              continue;

            // 1  3  5
            if (nghbs[1] != minV && nghbs[3] != minV && nghbs[5] != minV)
              continue;
            // 3  5  7
            if (nghbs[3] != minV && nghbs[5] != minV && nghbs[7] != minV)
              continue;

            bufMod[pixOffset] = minV;
            modified          = true;
            done              = false;
          }
        }
        if (modified)
          copy(imMod, imTmp);

        // second sub-iteration
        modified = false;
        for (off_t y = 0; y < height; y++) {
          off_t lineOffset = y * width;

#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
          for (off_t x = 0; x < width; x++) {
            off_t     pixOffset = lineOffset + x;

            if (bufTmp[pixOffset] == minV)
              continue;

            T nghbs[8];
            getNeighborhood(imTmp, x, y, nghbs, Xr, Bp);
            if (Bp < 2 || Bp > 6)
              continue;
            if (Xr != 2)
              continue;

            //  1  3  7
            if (nghbs[1] != minV && nghbs[3] != minV && nghbs[7] != minV)
              continue;
            //  1  5  7
            if (nghbs[1] != minV && nghbs[5] != minV && nghbs[7] != minV)
              continue;

            bufMod[pixOffset] = minV;
            modified          = true;
            done              = false;
          }
        }
        if (modified)
          copy(imMod, imTmp);

      } while (!done);

      copy(imTmp, imOut);

      return RES_OK;
    }
  };
  /** @endcond */

  /**
   * imageThinning()
   *
   * @param[in] imIn : binary input image
   * @param[out] imOut : output image
   * @param[in] method : algorithm to use.
   * - Zhang (default) - @cite zhang_suen_1984
   * - DongLinHuang - @cite dong_lin_huang_2016
   */
  template <typename T>
  RES_T imageThinning(const Image<T> &imIn, Image<T> &imOut, string method = "Zhang")
  {
    if (method == "Zhang") {
        return zhangSkeleton(imIn, imOut);
    }
    if (method == "DongLinHuang") {
        ZhangThinning<T> zt;
        return zt.skDongLinHuang(imIn, imOut);
    }
    ERR_MSG("Method not implemented : " + method);
    return RES_ERR;
  }

  /** @} */

} // namespace smil

#endif // _D_SKELETON_HPP
