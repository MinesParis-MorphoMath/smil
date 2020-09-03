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
   *  - soillebook{Chap. 5}
   *
   * @{
   */

  /**
   * skiz() - Skeleton by Influence Zones @txtbold{(Skiz)}
   *
   * Thinning of the background with a Composite Structuring Element
   * @b HMT_sL(6) followed by a thinning with a Composite Structuring Element
   * @b HMT_hM(6).
   *
   * @see
   * - @soillebook{p. 170}
   * - @b HMT_sL() and @b HMT_hM() in CompStrEltList
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
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
   * Extinction values
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
   * Implementation corresponding to the algorithm described in
   * @cite khanyile_comparative_2011.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
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

  struct OffsetStruct {
    off_t x, y, z;
    off_t o;
    off_t w, h, d;

    OffsetStruct(size_t S[3])
    {
      w = S[0];
      h = S[1];
      d = S[2];
      x = y = z = o = 0;
    }

    OffsetStruct(const OffsetStruct &offset)
    {
      w = offset.w;
      h = offset.h;
      d = offset.d;
      x = offset.x;
      y = offset.y;
      z = offset.z;
      o = offset.o;
    }

    OffsetStruct(size_t w, size_t h, size_t d = 1)
    {
      this->w = w;
      this->h = h;
      this->d = d;
      x = y = z = o = 0;
    }

    void setCoords(off_t x, off_t y, off_t z = 0)
    {
      this->x = x;
      this->y = y;
      this->z = z;
      this->o = x + y * w + z * w * h;
    }

    void setOffset(off_t offset)
    {
      this->o = offset;

      this->x = offset % w;
      offset  = (offset - this->x) / w;
      this->y = offset % h;
      this->z = (offset - this->y) / h;
    }

    IntPoint getPoint()
    {
      IntPoint pt(x, y, z);
      return pt;
    }

    off_t getOffset()
    {
      return o;
    }

    void shift(off_t dx, off_t dy, off_t dz)
    {
      x += dx;
      y += dy;
      z += dz;
      this->o = x + y * w + z * w * h;
    }

    void shift(IntPoint p)
    {
      x += p.x;
      y += p.y;
      z += p.z;
      this->o = x + y * w + z * w * h;
    }

    bool pointInWindow(off_t x, off_t y, off_t z)
    {
      return (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d);
    }

    bool inImage()
    {
      return (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d);
    }
  };

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
    T *in  = imIn.getPixels();
    T *out = imOut.getPixels();

    fill<T>(imOut, T(0));

    size_t S[3];
    imIn.getSize(S);

    size_t nbrPixels   = S[0] * S[1] * S[2];
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
        OffsetStruct pt(S);
        pt.setOffset(i);

        bool up   = false;
        bool down = false;
        if (in[pt.o] > 0 && in[pt.o] != ImDtTypes<T>::max()) {
          for (UINT pts = 0; pts < sePtsNumber; ++pts) {
            OffsetStruct qt(S);
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

  template <class T>
  RES_T pruneSkizOld(const Image<T> &imIn, Image<T> &imOut, const StrElt &se)
  {
    T *in  = imIn.getPixels();
    T *out = imOut.getPixels();

    struct index {
      size_t x;
      size_t y;
      size_t z;
      size_t o;
      index(){};
    };

    OffsetStruct idx;

    fill<T>(imOut, T(0));

    size_t S[3];
    imIn.getSize(S);
    size_t nbrPixelsInSlice = S[0] * S[1];
    size_t nbrPixels        = nbrPixelsInSlice * S[2];
    // StrElt se               = se;
    UINT sePtsNumber = se.points.size();

#ifdef USE_OPEN_MP
    UINT nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel num_threads(nthreads)
#endif
    {
      index p, q;
      UINT pts;
      bool up, down;

#ifdef USE_OPEN_MP
#pragma omp for
#endif
      for (size_t i = 0; i < nbrPixels; ++i) {
        p.o = i;
        p.z = p.o / nbrPixelsInSlice;
        p.y = (p.o % nbrPixelsInSlice) / S[0];
        p.x = p.o % S[0];

        up   = false;
        down = false;
        if (in[p.o] > 0 && in[p.o] != ImDtTypes<T>::max()) {
          p.z = p.o / nbrPixelsInSlice;
          p.y = (p.o % nbrPixelsInSlice) / S[0];
          p.x = p.o % S[0];
          for (pts = 0; pts < sePtsNumber; ++pts) {
            q.x = p.x + se.points[pts].x;
            q.y = p.y + se.points[pts].y;
            q.z = p.z + se.points[pts].z;
            if (q.x < S[0] && q.y < S[1] && q.z < S[2]) {
              q.o = q.x + q.y * S[0] + q.z * nbrPixelsInSlice;

              if (in[q.o] != ImDtTypes<T>::max()) {
                if (in[q.o] >= in[p.o] + 1) {
                  up = true;
                }
                if (in[q.o] <= in[p.o] - 1) {
                  down = true;
                }
              }
            }
          }

          if (!up || !down) {
            out[p.o] = in[p.o];
          }
        }
      }
    }

    return RES_OK;
  }

  /** @} */

} // namespace smil

#endif // _D_SKELETON_HPP
