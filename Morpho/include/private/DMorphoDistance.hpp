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

#ifndef _D_MORPHO_DISTANCE_HPP
#define _D_MORPHO_DISTANCE_HPP

#include "DMorphImageOperations.hpp"
#include "DMorphoHierarQ.hpp"
#include "Base/include/private/DImageDraw.hpp"
#include "Base/include/private/DImageHistogram.hpp"
#include "Morpho/include/private/DMorphoBase.hpp"

namespace smil
{
  /** @{ */
  /** @cond */
  /*
   * @ingroup Morpho
   * @defgroup Distance Distance Function
   * @{
   */

  /*
   * Distance Functor
   */
  class DistanceFunctor
  {
  public:
    DistanceFunctor(StrElt se)
    {
      this->se = se;
    }

    DistanceFunctor()
    {
      this->se = DEFAULT_SE;
    }

    ~DistanceFunctor()
    {
    }

    template <class T1, class T2>
    RES_T distance(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se)
    {
      int st   = se.getType();
      int size = se.size;

      if (size > 1)
        return distGeneric(imIn, imOut, se);
      switch (st) {
        case SE_Cross:
          return distCross(imIn, imOut);
        case SE_Cross3D:
          return distCross3d(imIn, imOut);
        case SE_Squ:
          return distSquare(imIn, imOut);
        default:
          return distGeneric(imIn, imOut, se);
      }
      return RES_ERR;
    }

  private:
    StrElt se;

    /*
     * Check if a point is inside image bounds
     */
    bool ptInImage(off_t x, off_t y, off_t z, size_t *size)
    {
      if (x < 0 || x >= (off_t) size[0])
        return false;
      if (y < 0 || y >= (off_t) size[1])
        return false;
      if (z < 0 || z >= (off_t) size[2])
        return false;
      return true;
    }

    /*
     * Generic Distance function.
     */
    template <class T1, class T2>
    RES_T distGeneric(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);

      typedef Image<T1>                       imageInType;
      typedef typename imageInType::lineType  lineInType;
      typedef Image<T2>                       imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType  pixelsIn;
      lineOutType pixelsOut = imOut.getPixels();

      Image<T1> tmp(imIn);
      Image<T1> tmp2(imIn);

      // Set image to 1 when pixels are != 0
      ASSERT(inf(imIn, T1(1), tmp) == RES_OK);
      // ASSERT(mul(tmp, T1(255), tmp) == RES_OK);

      // Demi-Gradient to remove sources inside cluster of sources.
      ASSERT(erode(tmp, tmp2, se) == RES_OK);
      ASSERT(sub(tmp, tmp2, tmp) == RES_OK);

      ASSERT(copy(tmp, imOut) == RES_OK);

      queue<size_t> *level      = new queue<size_t>();
      queue<size_t> *next_level = new queue<size_t>();
      queue<size_t> *swap;
      T2             cur_level = T2(2);

      size_t size[3];
      imIn.getSize(size);
      // pixels per line
      size_t ppLine = size[0];
      // pixels per slice
      size_t ppSlice = size[0] * size[1];

      pixelsIn = imIn.getPixels();

      for (off_t i = 0, iMax = imIn.getPixelCount(); i < iMax; ++i) {
        if (pixelsOut[i] > T2(0)) {
          level->push(i);
          pixelsOut[i] = T2(1);
        }
      }

      off_t cur;
      off_t x, y, z, n_x, n_y, n_z;

      vector<IntPoint>           sePoints = se.points;
      vector<IntPoint>::iterator pt;

      bool oddLine;

      do {
        while (!level->empty()) {
          cur = level->front();
          pt  = sePoints.begin();

          z = cur / ppSlice;
          y = (cur - z * ppSlice) / ppLine;
          x = cur - y * ppLine - z * ppSlice;

          oddLine = se.odd && (y % 2);

          while (pt != sePoints.end()) {
            n_x = x + pt->x;
            n_y = y + pt->y;
            n_x += (oddLine && ((n_y + 1) % 2) != 0) ? 1 : 0;
            n_z = z + pt->z;

            off_t offset = n_x + n_y * ppLine + n_z * ppSlice;

            // if (n_x >= 0 && n_x < (int) ppLine && n_y >= 0 &&
            //    n_y < (int) size[1] && n_z >= 0 && n_z < (int) size[2] &&
            if (ptInImage(n_x, n_y, n_z, size) && pixelsOut[offset] == T2(0) &&
                pixelsIn[offset] > T1(0)) {
              pixelsOut[offset] = T2(cur_level);
              next_level->push(offset);
            }
            ++pt;
          }
          level->pop();
        }
        ++cur_level;

        swap       = level;
        level      = next_level;
        next_level = swap;
      } while (!level->empty());

      delete level;
      delete next_level;

      return RES_OK;
    }

    /*
     * Distance Cross3D function (???).
     */
    template <class T1, class T2>
    RES_T distCross3d(const Image<T1> &imIn, Image<T2> &imOut)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);
      Image<T1>    tmp(imIn);
      ASSERT(inf(imIn, T1(1), tmp) == RES_OK);

      typedef Image<T1>                       imageInType;
      typedef typename imageInType::lineType  lineInType;
      typedef Image<T2>                       imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType  pixelsIn  = tmp.getPixels();
      lineOutType pixelsOut = imOut.getPixels();

      int size[3];
      imIn.getSize(size);
      size_t   offset;
      long int x, y, z;
      T2       infinite = ImDtTypes<T2>::max();
      long int min;

      for (z = 0; z < size[2]; ++z) {
#ifdef USE_OPEN_MP
#pragma omp for private(offset, x, y, min)
#endif // USE_OPEN_MP
        for (x = 0; x < size[0]; ++x) {
          offset = z * size[1] * size[0] + x;
          if (pixelsIn[offset] == T1(0)) {
            pixelsOut[offset] = T2(0);
          } else {
            pixelsOut[offset] = infinite;
          }

          for (y = 1; y < size[1]; ++y) {
            if (pixelsIn[offset + y * size[0]] == T1(0)) {
              pixelsOut[offset + y * size[0]] = T2(0);
            } else {
              pixelsOut[offset + y * size[0]] =
                  (1 + pixelsOut[offset + (y - 1) * size[0]] > infinite)
                      ? infinite
                      : 1 + pixelsOut[offset + (y - 1) * size[0]];
            }
          }

          for (y = size[1] - 2; y >= 0; --y) {
            min = (pixelsOut[offset + (y + 1) * size[0]] + 1 > infinite)
                      ? infinite
                      : pixelsOut[offset + (y + 1) * size[0]] + 1;
            if (min < pixelsOut[offset + y * size[0]])
              pixelsOut[offset + y * size[0]] =
                  (1 + pixelsOut[offset + (y + 1) * size[0]]);
          }
        }

#ifdef USE_OPEN_MP
#pragma omp for private(x, y, offset)
#endif // USE_OPEN_MP
        for (y = 0; y < size[1]; ++y) {
          offset = z * size[1] * size[0] + y * size[0];
          for (x = 1; x < size[0]; ++x) {
            if (pixelsOut[offset + x] != 0 &&
                pixelsOut[offset + x] > pixelsOut[offset + x - 1]) {
              pixelsOut[offset + x] = pixelsOut[offset + x - 1] + 1;
            }
          }
          for (x = size[0] - 2; x >= 0; --x) {
            if (pixelsOut[offset + x] != 0 &&
                pixelsOut[offset + x] > pixelsOut[offset + x + 1]) {
              pixelsOut[offset + x] = pixelsOut[offset + x + 1] + 1;
            }
          }
        }
      }
      for (y = 0; y < size[1]; ++y) {
#ifdef USE_OPEN_MP
#pragma omp for private(x, z, offset)
#endif // USE_OPEN_MP
        for (x = 0; x < size[0]; ++x) {
          offset = y * size[0] + x;
          for (z = 1; z < size[2]; ++z) {
            if (pixelsOut[offset + z * size[1] * size[0]] != 0 &&
                pixelsOut[offset + z * size[1] * size[0]] >
                    pixelsOut[offset + (z - 1) * size[1] * size[0]]) {
              pixelsOut[offset + z * size[1] * size[0]] =
                  pixelsOut[offset + (z - 1) * size[1] * size[0]] + 1;
            }
          }
          for (z = size[2] - 2; z >= 0; --z) {
            if (pixelsOut[offset + z * size[1] * size[0]] != 0 &&
                pixelsOut[offset + z * size[1] * size[0]] >
                    pixelsOut[offset + (z + 1) * size[1] * size[0]]) {
              pixelsOut[offset + z * size[1] * size[0]] =
                  pixelsOut[offset + (z + 1) * size[1] * size[0]] + 1;
            }
          }
        }
      }
      return RES_OK;
    }

    /*
     * Distance Cross function (???).
     */
    template <class T1, class T2>
    RES_T distCross(const Image<T1> &imIn, Image<T2> &imOut)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);
      Image<T1>    tmp(imIn);
      ASSERT(inf(imIn, T1(1), tmp) == RES_OK);

      typedef Image<T1>                       imageInType;
      typedef typename imageInType::lineType  lineInType;
      typedef Image<T2>                       imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType  pixelsIn  = tmp.getPixels();
      lineOutType pixelsOut = imOut.getPixels();

      off_t size[3];
      imIn.getSize(size);
      off_t offset;
      off_t x, y, z;

      T2 infinite = ImDtTypes<T2>::max();

      for (z = 0; z < (off_t) size[2]; ++z) {
#ifdef USE_OPEN_MP
#pragma omp for private(offset, x, y)
#endif // USE_OPEN_MP
        for (x = 0; x < (off_t) size[0]; ++x) {
          offset = z * size[1] * size[0] + x;
          if (pixelsIn[offset] == T1(0)) {
            pixelsOut[offset] = T2(0);
          } else {
            pixelsOut[offset] = infinite;
          }

          for (y = 1; y < (off_t) size[1]; ++y) {
            if (pixelsIn[offset + y * size[0]] == T1(0)) {
              pixelsOut[offset + y * size[0]] = T2(0);
            } else {
              // pixelsOut[offset + y * size[0]] =
              //    (1 + pixelsOut[offset + (y - 1) * size[0]] > infinite)
              //        ? infinite
              //        : 1 + pixelsOut[offset + (y - 1) * size[0]];
              if (pixelsOut[offset + (y - 1) * size[0]] < infinite)
                pixelsOut[offset + y * size[0]] =
                    pixelsOut[offset + (y - 1) * size[0]] + 1;
              else
                pixelsOut[offset + y * size[0]] = infinite;
            }
          }

          for (y = (off_t) size[1] - 2; y >= 0; --y) {
            T2 minVal;
            // minVal = (pixelsOut[offset + (y + 1) * size[0]] + 1 > infinite)
            //          ? infinite
            //          : pixelsOut[offset + (y + 1) * size[0]] + 1;

            minVal = pixelsOut[offset + (y + 1) * size[0]];
            if (minVal < infinite)
              minVal = pixelsOut[offset + (y + 1) * size[0]] + 1;
            if (minVal < pixelsOut[offset + y * size[0]])
              pixelsOut[offset + y * size[0]] =
                  (pixelsOut[offset + (y + 1) * size[0]] + 1);
          }
        }

#ifdef USE_OPEN_MP
#pragma omp for private(x, y, offset)
#endif // USE_OPEN_MP
        for (y = 0; y < (off_t) size[1]; ++y) {
          offset = z * size[1] * size[0] + y * size[0];
          for (x = 1; x < (off_t) size[0]; ++x) {
            if (pixelsOut[offset + x] != 0 &&
                pixelsOut[offset + x] > pixelsOut[offset + x - 1]) {
              pixelsOut[offset + x] = pixelsOut[offset + x - 1] + 1;
            }
          }
          for (x = (off_t) size[0] - 2; x >= 0; --x) {
            if (pixelsOut[offset + x] != 0 &&
                pixelsOut[offset + x] > pixelsOut[offset + x + 1]) {
              pixelsOut[offset + x] = pixelsOut[offset + x + 1] + 1;
            }
          }
        }
      }
      return RES_OK;
    }

    /*
     * Distance Square function (???).
     */
    template <class T1, class T2>
    RES_T distSquare(const Image<T1> &imIn, Image<T2> &imOut)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);
      Image<T2>    tmp(imIn);

      typedef Image<T1>                       imageInType;
      typedef typename imageInType::lineType  lineInType;
      typedef Image<T2>                       imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType  pixelsIn  = imIn.getPixels();
      lineOutType pixelsOut = imOut.getPixels();
      lineOutType pixelsTmp = tmp.getPixels();

      off_t size[3];
      imIn.getSize(size);
      off_t offset;
      off_t x, y, z;
      T2    infinite = ImDtTypes<T2>::max();

      uint32_t minV;

      // H(x,u) is a minimizer,
      // = MIN(h: 0 <= h < u & Any (i: 0 <= i < u : f(x,h) <= f(x,i)) : h )
      size_t size_array = MAX(size[0], size[1]);
      // sets of the least minimizers that occurs
      // during the scan from left to right.
      vector<off_t> s(size_array);
      // sets of points with the same least minimizer
      vector<off_t> t(size_array);
      s[0]    = 0;
      t[0]    = 0;
      off_t q = 0;
      off_t w;

      uint32_t tmpdist, tmpdist2;

      for (z = 0; z < size[2]; ++z) {
#ifdef USE_OPEN_MP
#pragma omp for private(offset, x, y, minV)
#endif // USE_OPEN_MP
        for (x = 0; x < size[0]; ++x) {
          offset = z * size[1] * size[0] + x;
          if (pixelsIn[offset] == T1(0)) {
            pixelsTmp[offset] = T2(0);
          } else {
            pixelsTmp[offset] = infinite;
          }
          // SCAN 1
          for (y = 1; y < size[1]; ++y) {
            if (pixelsIn[offset + y * size[0]] == T1(0)) {
              pixelsTmp[offset + y * size[0]] = T2(0);
            } else {
              // pixelsTmp[offset + y * size[0]] =
              //     (1 + pixelsTmp[offset + (y - 1) * size[0]] > infinite)
              //         ? infinite
              //         : 1 + pixelsTmp[offset + (y - 1) * size[0]];
              if (pixelsTmp[offset + (y - 1) * size[0]] < infinite) {
                pixelsTmp[offset + y * size[0]] =
                    pixelsTmp[offset + (y - 1) * size[0]] + 1;
              } else {
                pixelsTmp[offset + y * size[0]] = infinite;
              }
            }
          }
          // SCAN 2
          for (y = size[1] - 2; y >= 0; --y) {
            // minV = (pixelsTmp[offset + (y + 1) * size[0]] + 1 > infinite)
            //           ? infinite
            //           : pixelsTmp[offset + (y + 1) * size[0]] + 1;
            minV = pixelsTmp[offset + (y + 1) * size[0]];
            if (minV < infinite)
              minV = pixelsTmp[offset + (y + 1) * size[0]] + 1;

            if (minV < pixelsTmp[offset + y * size[0]])
              pixelsTmp[offset + y * size[0]] =
                  (1 + pixelsTmp[offset + (y + 1) * size[0]]);
          }
        }

#ifdef USE_OPEN_MP
#pragma omp for private(offset, y, q, w, tmpdist, tmpdist2)
#endif // USE_OPEN_MP
        for (y = 0; y < size[1]; ++y) {
          offset = z * size[1] * size[0] + y * size[0];
          q      = 0;
          t[0]   = 0;
          s[0]   = 0;
          // SCAN 3
          for (off_t u = 1; u < size[0]; ++u) {
#if 1
            do {
              tmpdist = std::abs(t[q] - s[q]);
              if (tmpdist < pixelsTmp[offset + s[q]])
                tmpdist = pixelsTmp[offset + s[q]];

              tmpdist2 = std::abs(t[q] - u);
              if (tmpdist2 < pixelsTmp[offset + u])
                tmpdist2 = pixelsTmp[offset + u];
              if (tmpdist > tmpdist2)
                q--;
            } while (q >= 0 && tmpdist > tmpdist2);
#else
            // tmpdist = (t[q] > s[q]) ? t[q] - s[q] : s[q] - t[q];
            // tmpdist = (tmpdist >= pixelsTmp[offset + s[q]])
            //              ? tmpdist
            //              : pixelsTmp[offset + s[q]];

            tmpdist = std::abs(t[q] - s[q]);
            if (tmpdist < pixelsTmp[offset + s[q]])
              tmpdist = pixelsTmp[offset + s[q]];

            // tmpdist2 = (t[q] > u) ? t[q] - u : u - t[q];
            // tmpdist2 = (tmpdist2 >= pixelsTmp[offset + u])
            //               ? tmpdist2
            //               : pixelsTmp[offset + u];

            tmpdist2 = std::abs(t[q] - u);
            if (tmpdist2 < pixelsTmp[offset + u])
              tmpdist2 = pixelsTmp[offset + u];

            while (q >= 0 && tmpdist > tmpdist2) {
              q--;
              if (q >= 0) {
                tmpdist = (t[q] > s[q]) ? t[q] - s[q] : s[q] - t[q];
                tmpdist = (tmpdist >= pixelsTmp[offset + s[q]])
                              ? tmpdist
                              : pixelsTmp[offset + s[q]];
                tmpdist2 = (t[q] > u) ? t[q] - u : u - t[q];
                tmpdist2 = (tmpdist2 >= pixelsTmp[offset + u])
                               ? tmpdist2
                               : pixelsTmp[offset + u];
              }
            }
#endif
            if (q < 0) {
              q    = 0;
              s[0] = u;
            } else {
              if (pixelsTmp[offset + s[q]] <= pixelsTmp[offset + u]) {
#if 0
                w = (s[q] + pixelsTmp[offset + u] >= (s[q] + u) / 2)
                        ? s[q] + pixelsTmp[offset + u]
                        : (s[q] + u) / 2;
#else
                w = s[q] + pixelsTmp[offset + u];
                if (w < (s[q] + u) / 2)
                  w = (s[q] + u) / 2;
#endif
              } else {
#if 0
                w = (u - pixelsTmp[offset + s[q]] >= (s[q] + u) / 2)
                        ? (s[q] + u) / 2
                        : u - pixelsTmp[offset + s[q]];
#else
                w = u - pixelsTmp[offset + s[q]];
                if (w >= (s[q] + u) / 2)
                  w = (s[q] + u) / 2;
#endif
              }
              w = 1 + w;
              if (w < size[0]) {
                q++;
                s[q] = u;
                t[q] = w;
              }
            }
          }

          // SCAN 4
          for (off_t u = size[0] - 1; u >= 0; --u) {
#if 0
            pixelsOut[offset + u] = (u > s[q]) ? u - s[q] : s[q] - u;
            pixelsOut[offset + u] =
                (pixelsOut[offset + u] >= pixelsTmp[offset + s[q]])
                    ? pixelsOut[offset + u]
                    : pixelsTmp[offset + s[q]];
#else
            pixelsOut[offset + u] = std::abs(u - s[q]);
            if (pixelsOut[offset + u] < pixelsTmp[offset + s[q]])
              pixelsOut[offset + u] = pixelsTmp[offset + s[q]];
#endif
            if (u == t[q])
              q--;
          }
        }
      }
      return RES_OK;
    }
  };

  /*
   * Wrapper to DistanceFunctor
   */
  template <class T1, class T2>
  RES_T distance(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se)
  {
    DistanceFunctor df;

    return df.distance(imIn, imOut, se);
  }

  /*
   * Geodesic Distance Function
   */
  //
  //  ####   ######   ####   #####   ######   ####      #     ####
  // #    #  #       #    #  #    #  #       #          #    #    #
  // #       #####   #    #  #    #  #####    ####      #    #
  // #  ###  #       #    #  #    #  #            #     #    #
  // #    #  #       #    #  #    #  #       #    #     #    #    #
  //  ####   ######   ####   #####   ######   ####      #     ####
  //
  template <class T1, class T2>
  RES_T distanceGeodesic(const Image<T1> &imIn, const Image<T1> &imMask,
                         Image<T2> &imOut, const StrElt &se)
  {
    ASSERT_ALLOCATED(&imIn, &imOut, &imMask);
    ASSERT_SAME_SIZE(&imIn, &imOut, &imMask);

    ImageFreezer freeze(imOut);

    Image<UINT32> imOffset(imIn);

    typedef Image<T1>                       imageInType;
    typedef typename imageInType::lineType  lineInType;
    typedef Image<T2>                       imageOutType;
    typedef typename imageOutType::lineType lineOutType;

    lineInType  pixelsIn, pixelsMask;
    lineOutType pixelsOut = imOut.getPixels();

    Image<T1> tmp(imIn);

    vector<IntPoint> sePts;
    off_t            sePtsNbr;

    typedef size_t                 OffsetT;
    HierarchicalQueue<T2, OffsetT> hq;
    // T3D fah      PriorityQueueT<float, OffsetT> hq(true);
    // Set image to 1 when pixels are !=0
    ASSERT(inf(imIn, T1(1), tmp) == RES_OK);
    ASSERT(mul(tmp, T1(255), tmp) == RES_OK);

    pixelsIn   = imIn.getPixels();
    pixelsMask = imMask.getPixels();

    // ...... Structuring element manipulation
    sePts.clear();

    // set an offset distance for each se point (!=0,0,0)
    vector<IntPoint>::const_iterator it;
    for (it = se.points.begin(); it != se.points.end(); it++) {
      if (it->x != 0 || it->y != 0 || it->z != 0) {
        sePts.push_back(*it);
      }
    }
    sePtsNbr = sePts.size();
    // ...... END Structuring element manipulation

    T2 T2_maxVal = ImDtTypes<T2>::max();

    off_t x0, y0, z0;
    off_t x1, y1, z1;
    off_t x2, y2, z2;

    off_t p0, p1, p2;

    float current_dist, wd;

    off_t imSize[3];
    imIn.getSize(imSize);
    off_t              maxOffset = imSize[2] * imSize[1] * imSize[0];
    std::vector<float> distVector(maxOffset, ImDtTypes<float>::max());

    // INITIALIZE
    // Pixels != 0 but with a 0 neighbor (contour pixels) are:
    //  - pushed in the FAH
    //  - out[p] = 1
    //  - pixelOffset[p]=bckg neighbor

    // pixelOffset: contains the closest bck pixel for each object pixel (for
    // pixels with computed distance value)
    //  maxOffset+1 for pixels not processed yet (or background pixels)
    hq.initialize(imIn);
    for (off_t i = 0; i < maxOffset; ++i) {
      if ((pixelsIn[i] > T1(0)) && (pixelsMask[i] > 0)) {
        imIn.getCoordsFromOffset(i, x0, y0, z0);
        //    std::cout<<"PROCESS:"<<x0<<","<<y0<<"\n";
        bool oddLine = se.odd && ((y0) % 2);

        for (off_t k = 0; k < sePtsNbr; k++) {
          IntPoint &pt = sePts[k];
          x1           = x0 + pt.x;
          y1           = y0 + pt.y;
          z1           = z0 + pt.z;

          if (oddLine) {
            x1 += (((y1 + 1) % 2) != 0);
          }

          if (x1 >= 0 && x1 < imSize[0] && y1 >= 0 && y1 < imSize[1] &&
              z1 >= 0 && z1 < imSize[2]) {
            p1 = x1 + (y1) *imSize[0] + (z1) *imSize[1] * imSize[0];

            if ((pixelsIn[p1] == 0) && (pixelsMask[p1] > 0)) {
              hq.push(T2(1), i);
              //    std::cout<<"INIT push:"<<x0<<","<<y0<<"\n";
              pixelsOut[i] = 1;
              break;
            }
          }  // if p1 in image
        }    // for each ngb
      }      // if px in object
      else { // NULL Pixel
        pixelsOut[i] = 0;
      }
    } // for each pixel

    //#####################################

    // Process hierarchical queue. Get a pixel(p0), find ngb without
    // computed distance(p1), p2 is a ngb of p1. If p2 knows its
    // distance, find p3 (the closest bckg pixel to p2) and compute
    // distance (p1,p3). Keep the smallest distance to p1 and assign
    // it. The origin of p1 is set, pixelsOffset[p1] is set to oo
    // (the p3 leading to the smallest distance).

    while (!hq.isEmpty()) {
      p0 = hq.pop();
      imIn.getCoordsFromOffset(p0, x0, y0, z0);
      if (pixelsOut[p0] == 0) {
        std::cout << "BAD PLACE, BAD TIME." << int(p0)
                  << "goes out from Queue wo assigned distance\n";
      }

      bool oddLine = se.odd && ((y0) % 2);

      for (off_t i = 0; i < (off_t ) sePtsNbr; i++) {
        IntPoint &pt = sePts[i];
        x1           = x0 + pt.x;
        y1           = y0 + pt.y;
        z1           = z0 + pt.z;

        if (oddLine) {
          x1 += (((y1 + 1) % 2) != 0);
        }

        if (x1 >= 0 && x1 < imSize[0] && y1 >= 0 && y1 < imSize[1] && z1 >= 0 &&
            z1 < imSize[2]) {
          p1 = x1 + (y1) *imSize[0] + (z1) *imSize[1] * imSize[0];

          if ((pixelsIn[p1] != 0) && (pixelsMask[p1] > 0) &&
              (pixelsOut[p1] == 0)) {
            // non NULL input pixel, but without output value
            //    (PixelsOffset == maxOffset+1)
            current_dist = ImDtTypes<float>::max(); // INFINITE!!!
            for (off_t k = 0; k < sePtsNbr; k++) {
              IntPoint &pt2 = sePts[k];
              x2            = x1 + pt2.x;
              y2            = y1 + pt2.y;
              z2            = z1 + pt2.z;

              bool oddLine2 = se.odd && ((y1) % 2);
              if (oddLine2) {
                x2 += (((y2 + 1) % 2) != 0);
              }

              if (x2 >= 0 && x2 < imSize[0] && y2 >= 0 && y2 < imSize[1] &&
                  z2 >= 0 && z2 < imSize[2]) {
                p2 = x2 + (y2) *imSize[0] + (z2) *imSize[1] * imSize[0];
                if (pixelsOut[p2] > 0) { //
                  /* voisin, p2, de distance calculee */

                  wd = pixelsOut[p2] + 1;

                  if (wd < current_dist) {
                    current_dist = wd;
                    // oo           = p2;
                  }
                } // if ngb knows its distance
              }   // if not border
            }     // for ngb

            float pr1 = current_dist;
            if (pr1 > T2_maxVal) {
              pr1 = T2_maxVal;
            } // saturation/clipping

            hq.push(pr1, p1);

            // pixelsOut[p1] = current_dist;
            pixelsOut[p1] = pr1;

          } // if wk[p_suiv] != DONE
        }   // x1 no border
      }     // for each ngb of p
    }       // while § EMPTY
    return RES_OK;
  } // END distanceGeodesic

  /*
   * Euclidean Distance function.
   */
  //
  // ######  #    #   ####   #          #    #####   ######    ##    #    #
  // #       #    #  #    #  #          #    #    #  #        #  #   ##   #
  // #####   #    #  #       #          #    #    #  #####   #    #  # #  #
  // #       #    #  #       #          #    #    #  #       ######  #  # #
  // #       #    #  #    #  #          #    #    #  #       #    #  #   ##
  // ######   ####    ####   ######     #    #####   ######  #    #  #    #
  //
  template <class T1, class T2>
  RES_T distanceEuclideanOld(const Image<T1> &imIn, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);
    Image<T2>    tmp(imIn);

    typedef Image<T1>                       imageInType;
    typedef typename imageInType::lineType  lineInType;
    typedef Image<T2>                       imageOutType;
    typedef typename imageOutType::lineType lineOutType;

    lineInType  pixelsIn  = imIn.getPixels();
    lineOutType pixelsOut = imOut.getPixels();
    lineOutType pixelsTmp = tmp.getPixels();

    int size[3];
    imIn.getSize(size);
    size_t nbrPixelsPerSlice = size[0] * size[1];
    size_t offset;
    int    x, y, z;
    T2     infinite = ImDtTypes<T2>::max();

    // H(x,u) is a minimizer, = MIN(h: 0 <= h < u & Any (i: 0 <= i < u : f(x,h)
    // <= f(x,i)) : h )
    // sets of the least minimizers that occurs
    // during the scan from left to right.
    vector<long int> s(size[0]);
    // sets of points with the same least minimizer
    vector<long int> t(size[0]);
    long int         q = 0;
    long int         w;

    for (z = 0; z < size[2]; ++z) {
      // #ifdef USE_OPEN_MP
      //   #pragma omp for private(offset,x,y,min)
      // #endif // USE_OPEN_MP
      for (x = 0; x < size[0]; ++x) {
        offset = z * nbrPixelsPerSlice + x;
        if (pixelsIn[offset] == T1(0)) {
          pixelsOut[offset] = T2(0);
        } else {
          pixelsOut[offset] = infinite;
        }
        // SCAN 1
        for (y = 1; y < size[1]; ++y) {
          if (pixelsIn[offset + y * size[0]] == T1(0)) {
            pixelsOut[offset + y * size[0]] = T2(0);
          } else {
            // BUG :
            // 1 + pixelsOut[offset + (y - 1) * size[0]]
            //    will never be greater than infinite
            pixelsOut[offset + y * size[0]] =
                (1 + pixelsOut[offset + (y - 1) * size[0]] > infinite)
                    ? infinite
                    : 1 + pixelsOut[offset + (y - 1) * size[0]];
          }
        }
        // SCAN 2
        for (y = size[1] - 2; y >= 0 && y < size[1]; --y) {
          if (pixelsOut[offset + (y + 1) * size[0]] <
              pixelsOut[offset + y * size[0]])
            pixelsOut[offset + y * size[0]] =
                (1 + pixelsOut[offset + (y + 1) * size[0]] < infinite)
                    ? 1 + pixelsOut[offset + (y + 1) * size[0]]
                    : infinite;
        }
      }
    }

    copy(imOut, tmp);

#define __f_euclidean(x, i)                                                    \
  (x - i) * (x - i) + pixelsTmp[offset + i] * pixelsTmp[offset + i]
#define __sep(a, b)                                                            \
  (b * b - a * a + pixelsTmp[offset + b] * pixelsTmp[offset + b] -             \
   pixelsTmp[offset + a] * pixelsTmp[offset + a]) /                            \
      (2 * (b - a))

    for (z = 0; z < size[2]; ++z) {
#ifdef USE_OPEN_MP
#pragma omp for private(offset, y, x, q, w)
#endif // USE_OPEN_MP
      for (y = 0; y < size[1]; ++y) {
        offset = z * nbrPixelsPerSlice + y * size[0];
        q      = 0;
        t[0]   = 0;
        s[0]   = 0;
        // SCAN 3
        for (x = 1; x < size[0]; ++x) {
          while (q >= 0 && __f_euclidean(t[q], s[q]) > __f_euclidean(t[q], x)) {
            q--;
          }
          if (q < 0) {
            q    = 0;
            s[0] = x;
          } else {
            w = 1 + __sep(s[q], x);
            if (w < size[0]) {
              q++;
              s[q] = x;
              t[q] = w;
            }
          }
        }
        // SCAN 4
        for (x = size[0] - 1; x >= 0 && x < size[0]; --x) {
          pixelsOut[offset + x] = __f_euclidean(x, s[q]);
          if (x == t[q])
            --q;
        }
      }
    }
#undef __f_euclidean
#undef __sep

    copy(imOut, tmp);

// The previous pixels are already squarred ...
#define __f_euclidean(a, i)                                                    \
  (a - i) * (a - i) + pixelsTmp[offset + i * nbrPixelsPerSlice]
#define __sep(a, b)                                                            \
  (b * b - a * a + pixelsTmp[offset + b * nbrPixelsPerSlice] -                 \
   pixelsTmp[offset + a * nbrPixelsPerSlice]) /                                \
      (2 * (b - a))

    for (y = 0; y < size[1]; ++y) {
#ifdef USE_OPENMP
#pragma omp for private(x, z, offset)
#endif
      for (x = 0; x < size[0]; ++x) {
        offset = y * size[0] + x;
        q      = 0;
        t[0]   = 0;
        s[0]   = 0;
        for (z = 1; z < size[2]; ++z) {
          while (q >= 0 && __f_euclidean(t[q], s[q]) > __f_euclidean(t[q], z)) {
            q--;
          }
          if (q < 0) {
            q    = 0;
            s[0] = z;
          } else {
            w = 1 + __sep(s[q], z);
            if (w < size[2]) {
              q++;
              s[q] = z;
              t[q] = w;
            }
          }
        }
        for (z = size[2] - 1; z >= 0 && z < size[2]; --z) {
          pixelsOut[offset + z * nbrPixelsPerSlice] = __f_euclidean(z, s[q]);
          if (z == t[q]) {
            --q;
          }
        }
      }
    }
#undef __f_euclidean
#undef __sep

#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
    for (size_t offset = 0; offset < imOut.getPixelCount(); offset++) {
      pixelsOut[offset] = T2(std::sqrt(pixelsOut[offset]));
    }

    return RES_OK;
  }
  /** @endcond */

  /** @cond */
  /*
   * Base distance function performed with successive erosions.
   */
  template <class T>
  RES_T distV0(const Image<T> &imIn, Image<T> &imOut, const StrElt &se)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn);

    // Set image to 1 when pixels are !=0
    ASSERT((inf(imIn, T(1), tmpIm) == RES_OK));

    ASSERT((copy(tmpIm, imOut) == RES_OK));

    do {
      ASSERT((erode(tmpIm, tmpIm, se) == RES_OK));
      ASSERT((add(tmpIm, imOut, imOut) == RES_OK));

    } while (vol(tmpIm) != 0);

    return RES_OK;
  }

  /** @endcond */

  /** @} */
} // namespace smil

#endif // _D_MORPHO_DISTANCE_HPP
