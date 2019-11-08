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
  /** @cond */
  /*
   * @ingroup Morpho
   * @defgroup Distance Distance Function
   * @{
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
     * Generic Distance function.
     */

    inline bool ptInImage(off_t x, off_t y, off_t z, size_t size[])
    {
      if (x < 0 || y < 0 || z < 0)
        return false;
      if (x < (off_t) size[0] && y < (off_t) size[1] && z < (off_t) size[2])
        return true;
      return false;
    }

    // OK
    template <class T1, class T2>
    RES_T distGeneric(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);

      typedef Image<T1> imageInType;
      typedef typename imageInType::lineType lineInType;
      typedef Image<T2> imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType pixelsIn   = imIn.getPixels();
      lineOutType pixelsOut = imOut.getPixels();

      Image<T1> tmp(imIn);
      Image<T1> tmp2(imIn);

      // Set image to 1 when pixels are !=0
      ASSERT(inf(imIn, T1(1), tmp) == RES_OK);
      ASSERT(mul(tmp, T1(255), tmp) == RES_OK);

      // Demi-Gradient to remove sources inside cluster of sources.
      ASSERT(erode(tmp, tmp2, se) == RES_OK);
      ASSERT(sub(tmp, tmp2, tmp) == RES_OK);
      ASSERT(copy(tmp, imOut) == RES_OK);

      size_t size[3];
      imIn.getSize(size);
      // pixels per line and per plan
      size_t ppLine = size[0];
      size_t ppPlan = size[0] * size[1];

      size_t pixCount = imIn.getPixelCount();

      queue<size_t> *level      = new queue<size_t>();
      queue<size_t> *next_level = new queue<size_t>();
      queue<size_t> *swap;
      T2 cur_level = T2(2);

      for (off_t i = 0, iMax = pixCount; i < iMax; ++i) {
        if (pixelsOut[i] > T2(0)) {
          level->push(i);
          pixelsOut[i] = T2(1);
        }
      }

      do {
        while (!level->empty()) {
          vector<IntPoint> sePoints = se.points;
          vector<IntPoint>::iterator pt;

          off_t cur;
          off_t x, y, z, n_x, n_y, n_z;
          bool oddLine;

          cur = level->front();
          pt  = sePoints.begin();

          z = cur / (ppPlan);
          y = (cur - z * ppPlan) / ppLine;
          x = cur - y * ppLine - z * ppPlan;

          oddLine = se.odd && (y % 2 != 0);

          while (pt != sePoints.end()) {
            n_x = x + pt->x;
            n_y = y + pt->y;
            n_z = z + pt->z;
            if (oddLine && ((n_y + 1) % 2) != 0)
              n_x += 1;
            // orig     : n_x += (oddLine && ((n_y + 1) % 2) != 0) ? 1 : 0;
            // better ? : n_x += (oddLine && ((n_y % 2) == 0) ? 1 : 0;

            off_t offset = n_x + n_y * ppLine + n_z * ppPlan;
            if (ptInImage(n_x, n_y, n_z, size) && pixelsOut[offset] == T2(0) &&
                pixelsIn[offset] > T1(0)) {
              pixelsOut[offset] = T2(cur_level);
              next_level->push(n_x + n_y * ppLine + n_z * ppPlan);
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

      return RES_OK;
    }

    /*
     * Distance Cross3D function (???).
     */
    // OK
    template <class T1, class T2>
    RES_T distCross3d(const Image<T1> &imIn, Image<T2> &imOut)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);
      Image<T1> tmp(imIn);
      ASSERT(inf(imIn, T1(1), tmp) == RES_OK);

      typedef Image<T1> imageInType;
      typedef typename imageInType::lineType lineInType;
      typedef Image<T2> imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType pixelsIn   = tmp.getPixels();
      lineOutType pixelsOut = imOut.getPixels();

      int size[3];
      imIn.getSize(size);
      // pixels per line and per plan
      size_t ppLine = size[0];
      size_t ppPlan = size[0] * size[1];

      // size_t pixCount = imIn.getPixelCount();

      off_t offset;
      off_t x, y, z;
      UINT64 infinite = ImDtTypes<T2>::max();
      UINT64 min;

      for (z = 0; z < size[2]; ++z) {
#ifdef USE_OPEN_MP
#pragma omp for private(offset, x, y, min)
#endif // USE_OPEN_MP
        for (x = 0; x < ppLine; ++x) {
          offset = z * ppPlan + x;
          if (pixelsIn[offset] == T1(0)) {
            pixelsOut[offset] = T2(0);
          } else {
            pixelsOut[offset] = infinite;
          }

          for (y = 1; y < size[1]; ++y) {
            off_t oy = offset + y * ppLine;
            if (pixelsIn[offset + y * ppLine] == T1(0)) {
              pixelsOut[offset + y * ppLine] = T2(0);
            } else {
              if ((UINT64) pixelsOut[oy - ppLine] + 1 > infinite)
                pixelsOut[oy] = infinite;
              else
                pixelsOut[oy] = pixelsOut[oy - ppLine] + 1;
              /*
               * pixelsOut[offset + y * ppLine] =
               *     (1 + pixelsOut[offset + (y - 1) * ppLine] > infinite)
               *       ? infinite
               *       : 1 + pixelsOut[offset + (y - 1) * ppLine];
               */
            }
          }

          for (y = size[1] - 2; y >= 0; --y) {
            off_t oy = offset + y * ppLine;
            if ((UINT64) pixelsOut[oy + ppLine] + 1 > infinite)
              min = infinite;
            else
              min = pixelsOut[oy + ppLine] + 1;
            if (min < pixelsOut[oy])
              pixelsOut[oy] = pixelsOut[oy + ppLine] + 1;

            /* 
             * min = (pixelsOut[offset + (y + 1) * ppLine] + 1 > infinite)
             *          ? infinite
             *          : pixelsOut[offset + (y + 1) * ppLine] + 1;
             * if (min < pixelsOut[offset + y * ppLine])
             *  pixelsOut[offset + y * ppLine] =
             *      (1 + pixelsOut[offset + (y + 1) * ppLine]);
             */
          }
        }

#ifdef USE_OPEN_MP
#pragma omp for private(x, y, offset)
#endif // USE_OPEN_MP
        for (y = 0; y < size[1]; ++y) {
          offset = z * ppPlan + y * ppLine;
          for (off_t x = 1; x < ppLine; ++x) {
            if (pixelsOut[offset + x] != 0 &&
                pixelsOut[offset + x] > pixelsOut[offset + x - 1]) {
              pixelsOut[offset + x] = pixelsOut[offset + x - 1] + 1;
            }
          }
          for (x = ppLine - 2; x >= 0; --x) {
            if (pixelsOut[offset + x] != 0 &&
                pixelsOut[offset + x] > pixelsOut[offset + x + 1] ) {
              pixelsOut[offset + x] = pixelsOut[offset + x + 1] + 1;
            }
          }
        }
      }
      for (y = 0; y < size[1]; ++y) {
#ifdef USE_OPEN_MP
#pragma omp for private(x, z, offset)
#endif // USE_OPEN_MP
        for (x = 0; x < ppLine; ++x) {
          offset = y * ppLine + x;
          for (z = 1; z < size[2]; ++z) {
            off_t oz = offset + z * ppPlan;
            if (pixelsOut[oz] != 0 && pixelsOut[oz] > pixelsOut[oz - ppPlan]) {
              pixelsOut[oz] = pixelsOut[oz - ppPlan] + 1;
            }
          }
          for (z = size[2] - 2; z >= 0; --z) {
            off_t oz = offset + z * ppPlan;
            if (pixelsOut[oz] != 0 && pixelsOut[oz] > pixelsOut[oz + ppPlan]) {
              pixelsOut[oz] = pixelsOut[oz + ppPlan] + 1;
            }
          }
        }
      }
      return RES_OK;
    }

    /*
     * Distance Cross function (???).
     */
    // OK
    template <class T1, class T2>
    RES_T distCross(const Image<T1> &imIn, Image<T2> &imOut)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);
      Image<T1> tmp(imIn);
      ASSERT(inf(imIn, T1(1), tmp) == RES_OK);

      typedef Image<T1> imageInType;
      typedef typename imageInType::lineType lineInType;
      typedef Image<T2> imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType pixelsIn   = tmp.getPixels();
      lineOutType pixelsOut = imOut.getPixels();

      int size[3];
      imIn.getSize(size);
      // pixels per line and per plan
      size_t ppLine = size[0];
      size_t ppPlan = size[0] * size[1];
      size_t depth  = size[2];

      size_t pixCount = imIn.getPixelCount();

      off_t offset;
      off_t x, y, z;
      UINT64 infinite = ImDtTypes<T2>::max();
      UINT64 min;

      for (z = 0; z < size[2]; ++z) {
#ifdef USE_OPEN_MP
#pragma omp for private(offset, x, y, min)
#endif // USE_OPEN_MP
        for (x = 0; x < ppLine; ++x) {
          offset = z * ppPlan + x;
          if (pixelsIn[offset] == T1(0)) {
            pixelsOut[offset] = T2(0);
          } else {
            pixelsOut[offset] = infinite;
          }

          for (y = 1; y < size[1]; ++y) {
            off_t oy = offset + y * ppLine;

            if (pixelsIn[oy] == T1(0)) {
              pixelsOut[oy] = T2(0);
            } else {
              UINT64 tl = (UINT64)(1 + pixelsOut[oy - ppLine]);
              if (tl > infinite)
                pixelsOut[oy] = infinite;
              else
                pixelsOut[oy] = tl;
              /*
              pixelsOut[offset + y * ppLine] =
                  (1 + pixelsOut[offset + (y - 1) * ppLine] > infinite)
                      ? infinite
                      : 1 + pixelsOut[offset + (y - 1) * ppLine];
              */
            }
          }

          for (y = size[1] - 2; y >= 0; --y) {
            off_t oy = offset + y * ppLine;
            UINT64 tv = (UINT64)(pixelsOut[oy + ppLine] + 1);
            if (tv > infinite)
              min = infinite;
            else
              min = tv;
            if (min < (UINT64) pixelsOut[oy])
              pixelsOut[oy] = (T2) tv;
            /*
            min = (pixelsOut[offset + (y + 1) * ppLine] + 1 > infinite)
                      ? infinite
                      : pixelsOut[offset + (y + 1) * ppLine] + 1;
            if (min < pixelsOut[offset + y * ppLine])
              pixelsOut[offset + y * ppLine] =
                  (1 + pixelsOut[offset + (y + 1) * ppLine]);
            */
          }
        }

#ifdef USE_OPEN_MP
#pragma omp for private(x, y, offset)
#endif // USE_OPEN_MP
        for (y = 0; y < size[1]; ++y) {
          offset = z * ppPlan + y * ppLine;
          for (x = 1; x < ppLine; ++x) {
            if (pixelsOut[offset + x] > pixelsOut[offset + x - 1]) {
              pixelsOut[offset + x] = pixelsOut[offset + x - 1] + 1;
            }
          }
          for (x = ppLine - 2; x >= 0; --x) {
            if (pixelsOut[offset + x] > pixelsOut[offset + x + 1]) {
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
    // in progress
    template <class T1, class T2>
    RES_T distSquare(const Image<T1> &imIn, Image<T2> &imOut)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);
      Image<T2> tmp(imIn);

      typedef Image<T1> imageInType;
      typedef typename imageInType::lineType lineInType;
      typedef Image<T2> imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType pixelsIn   = imIn.getPixels();
      lineOutType pixelsOut = imOut.getPixels();
      lineOutType pixelsTmp = tmp.getPixels();

      size_t size[3];
      imIn.getSize(size);
      size_t ppLine = size[0];
      size_t ppPlan = size[0] * size[1];
      size_t depth  = size[2];
      
      off_t offset;
      off_t x, y, z;
      UINT64 infinite = ImDtTypes<T2>::max();
      UINT64 min;

      // H(x,u) is a minimizer, 
      // MIN(h: 0 <= h < u & Any (i: 0 <= i < u : f(x,h) <= f(x,i)) : h )
      size_t size_array = MAX(size[0], size[1]);
      // sets of the least minimizers that occurs
      // during the scan from left to right.
      vector<size_t> s(size_array);
      // sets of points with the same least minimizer
      vector<size_t> t(size_array);
      s[0]       = 0;
      t[0]       = 0;
      long q = 0;
      long w;

      long int tmpdist, tmpdist2;

      for (z = 0; z < size[2]; ++z) {
#ifdef USE_OPEN_MP
#pragma omp for private(offset, x, y, min)
#endif // USE_OPEN_MP
        for (x = 0; x < ppLine; ++x) {
          offset = z * ppPlan + x;
          if (pixelsIn[offset] == T1(0)) {
            pixelsTmp[offset] = T2(0);
          } else {
            pixelsTmp[offset] = infinite;
          }
          // SCAN 1
          for (y = 1; y < size[1]; ++y) {
            off_t oy = offset + y * ppLine;
            if (pixelsIn[oy] == T1(0)) {
              pixelsTmp[oy] = T2(0);
            } else {
              if ((UINT64) pixelsTmp[oy - ppLine] + 1 > infinite)
                pixelsTmp[oy] = infinite;
              else
                pixelsTmp[oy] = 1 + pixelsTmp[oy - ppLine];
              /*
               * pixelsTmp[offset + y * ppLine] =
               *    (1 + pixelsTmp[offset + (y - 1) * ppLine] > infinite)
               *        ? infinite
               *        : 1 + pixelsTmp[offset + (y - 1) * ppLine];
               */
            }
          }
          // SCAN 2
          for (y = size[1] - 2; y >= 0; --y) {
            off_t oy = offset + y * ppLine;
            
            min = MIN((UINT64) pixelsTmp[oy + ppLine] + 1, infinite);
            if ((UINT64) pixelsTmp[oy + ppLine] + 1 > infinite)
              min = infinite;
            else
              min = pixelsTmp[oy + ppLine] + 1;
              
            if (min < pixelsTmp[oy])
              pixelsTmp[oy] = (pixelsTmp[oy + ppLine] + 1);
            /*
             * min = (pixelsTmp[offset + (y + 1) * ppLine] + 1 > infinite)
             *          ? infinite
             *          : pixelsTmp[offset + (y + 1) * ppLine] + 1;
             * if (min < pixelsTmp[offset + y * ppLine])
             *   pixelsTmp[offset + y * ppLine] =
             *       (1 + pixelsTmp[offset + (y + 1) * ppLine]);
             */
          }
        }
#ifdef USE_OPEN_MP
#pragma omp for private(offset, y, q, w, tmpdist, tmpdist2)
#endif // USE_OPEN_MP
        for (y = 0; y < size[1]; ++y) {
          offset = z * ppPlan + y * ppLine;
          q      = 0;
          t[0]   = 0;
          s[0]   = 0;
          // SCAN 3
          for (long u = 1; u < ppLine; ++u) {
            tmpdist = ABS(t[q] - s[q]);
            if (t[q] > s[q])
              tmpdist = t[q] - s[q];
            else
              tmpdist = s[q] - t[q];    
            tmpdist = MAX(tmpdist, pixelsTmp[offset + s[q]]);

            tmpdist2 = ABS(t[q] - u);
            if (t[q] > u)
              tmpdist2 = t[q] - u;
            else
              tmpdist2 = u - t[q];    
            tmpdist2 = MAX(tmpdist2, pixelsTmp[offset + u]);

            /*
            tmpdist = (t[q] > s[q]) ? t[q] - s[q] : s[q] - t[q];
            tmpdist = (tmpdist >= pixelsTmp[offset + s[q]])
                          ? tmpdist
                          : pixelsTmp[offset + s[q]];
            tmpdist2 = (t[q] > u) ? t[q] - u : u - t[q];
            tmpdist2 = (tmpdist2 >= pixelsTmp[offset + u])
                           ? tmpdist2
                           : pixelsTmp[offset + u];
            */
            while (q >= 0 && tmpdist > tmpdist2) {
              q--;
              if (q >= 0) {
                tmpdist = ABS(t[q] - s[q]);
                tmpdist = MAX(tmpdist, pixelsTmp[offset + s[q]]);

                tmpdist2 = ABS(t[q] - u);
                tmpdist2 = MAX(tmpdist2, pixelsTmp[offset + u]);

                /*
                tmpdist = (t[q] > s[q]) ? t[q] - s[q] : s[q] - t[q];
                tmpdist = (tmpdist >= pixelsTmp[offset + s[q]])
                              ? tmpdist
                              : pixelsTmp[offset + s[q]];
                tmpdist2 = (t[q] > u) ? t[q] - u : u - t[q];
                tmpdist2 = (tmpdist2 >= pixelsTmp[offset + u])
                               ? tmpdist2
                               : pixelsTmp[offset + u];
                */
              }
            }
            if (q < 0) {
              q    = 0;
              s[0] = u;
            } else {
              if (pixelsTmp[offset + s[q]] <= pixelsTmp[offset + u]) {
                w = MAX(s[q] + pixelsTmp[offset + u], (s[q] + u) / 2);
                /*
                w = (s[q] + pixelsTmp[offset + u] >= (s[q] + u) / 2)
                        ? s[q] + pixelsTmp[offset + u]
                        : (s[q] + u) / 2;
                */
              } else {
                w = MIN(u - pixelsTmp[offset + s[q]], (s[q] + u) / 2);
                /*
                w = (u - pixelsTmp[offset + s[q]] >= (s[q] + u) / 2)
                        ? (s[q] + u) / 2
                        : u - pixelsTmp[offset + s[q]];
                */
              }
              w = 1 + w;
              if (w < ppLine) {
                q++;
                s[q] = u;
                t[q] = w;
              }
            }
          }
          // SCAN 4
          for (int u = ppLine - 1; u >= 0; --u) {
            pixelsOut[offset + u] = MAX(u > s[q], s[q] - u);
            pixelsOut[offset + u] = MAX(pixelsOut[offset + u], pixelsTmp[offset + s[q]]);

            /*
            pixelsOut[offset + u] = (u > s[q]) ? u - s[q] : s[q] - u;
            pixelsOut[offset + u] =
                (pixelsOut[offset + u] >= pixelsTmp[offset + s[q]])
                    ? pixelsOut[offset + u]
                    : pixelsTmp[offset + s[q]];
            */
            if (u == t[q])
              q--;
          }
        }
      }
      return RES_OK;
    }
  };

  template <class T1, class T2>
  RES_T df_distance(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se)
  {
    DistanceFunctor df;

    return df.distance(imIn, imOut, se);
  }
  /*
   * end of Distance Functor
   */

  /*
   *  ####   #       #####
   * #    #  #       #    #
   * #    #  #       #    #
   * #    #  #       #    #
   * #    #  #       #    #
   *  ####   ######  #####
   */

  /*
   * Distance function
   */
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

    typedef Image<T1> imageInType;
    typedef typename imageInType::lineType lineInType;
    typedef Image<T2> imageOutType;
    typedef typename imageOutType::lineType lineOutType;

    lineInType pixelsIn;
    lineOutType pixelsOut = imOut.getPixels();

    Image<T1> tmp(imIn);
    Image<T1> tmp2(imIn);

    // Set image to 1 when pixels are !=0
    ASSERT(inf(imIn, T1(1), tmp) == RES_OK);
    ASSERT(mul(tmp, T1(255), tmp) == RES_OK);

    // Demi-Gradient to remove sources inside cluster of sources.
    ASSERT(erode(tmp, tmp2, se) == RES_OK);
    ASSERT(sub(tmp, tmp2, tmp) == RES_OK);

    ASSERT(copy(tmp, imOut) == RES_OK);

    queue<size_t> *level      = new queue<size_t>();
    queue<size_t> *next_level = new queue<size_t>();
    queue<size_t> *swap;
    T2 cur_level = T2(2);

    int size[3];
    imIn.getSize(size);
    // pixels per line
    // int ppl = size[0];
    // pixels per plan
    // int ppp = size[0] * size[1];

    pixelsIn = imIn.getPixels();

    for (off_t i = 0, iMax = size[2] * size[1] * size[0]; i < iMax; ++i) {
      if (pixelsOut[i] > T2(0)) {
        level->push(i);
        pixelsOut[i] = T2(1);
      }
    }

    size_t cur;
    long int x, y, z, n_x, n_y, n_z;

    vector<IntPoint> sePoints = se.points;
    vector<IntPoint>::iterator pt;

    bool oddLine;

    do {
      while (!level->empty()) {
        cur = level->front();
        pt  = sePoints.begin();

        z = cur / (size[1] * size[0]);
        y = (cur - z * size[1] * size[0]) / size[0];
        x = cur - y * size[0] - z * size[1] * size[0];

        oddLine = se.odd && (y % 2);

        while (pt != sePoints.end()) {
          n_x = x + pt->x;
          n_y = y + pt->y;
          n_x += (oddLine && ((n_y + 1) % 2) != 0) ? 1 : 0;
          n_z = z + pt->z;

          if (n_x >= 0 && n_x < (int) size[0] && n_y >= 0 &&
              n_y < (int) size[1] && n_z >= 0 && n_z < (int) size[2] &&
              pixelsOut[n_x + (n_y) *size[0] + (n_z) *size[1] * size[0]] ==
                  T2(0) &&
              pixelsIn[n_x + (n_y) *size[0] + (n_z) *size[1] * size[0]] >
                  T1(0)) {
            pixelsOut[n_x + (n_y) *size[0] + (n_z) *size[1] * size[0]] =
                T2(cur_level);
            next_level->push(n_x + (n_y) *size[0] + (n_z) *size[1] * size[0]);
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
    Image<T1> tmp(imIn);
    ASSERT(inf(imIn, T1(1), tmp) == RES_OK);

    typedef Image<T1> imageInType;
    typedef typename imageInType::lineType lineInType;
    typedef Image<T2> imageOutType;
    typedef typename imageOutType::lineType lineOutType;

    lineInType pixelsIn   = tmp.getPixels();
    lineOutType pixelsOut = imOut.getPixels();

    int size[3];
    imIn.getSize(size);
    size_t offset;
    long int x, y, z;
    T2 infinite = ImDtTypes<T2>::max();
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
    Image<T1> tmp(imIn);
    ASSERT(inf(imIn, T1(1), tmp) == RES_OK);

    typedef Image<T1> imageInType;
    typedef typename imageInType::lineType lineInType;
    typedef Image<T2> imageOutType;
    typedef typename imageOutType::lineType lineOutType;

    lineInType pixelsIn   = tmp.getPixels();
    lineOutType pixelsOut = imOut.getPixels();

    int size[3];
    imIn.getSize(size);
    size_t offset;
    long int x, y, z;
    T2 infinite = ImDtTypes<T2>::max();
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
    Image<T2> tmp(imIn);

    typedef Image<T1> imageInType;
    typedef typename imageInType::lineType lineInType;
    typedef Image<T2> imageOutType;
    typedef typename imageOutType::lineType lineOutType;

    lineInType pixelsIn   = imIn.getPixels();
    lineOutType pixelsOut = imOut.getPixels();
    lineOutType pixelsTmp = tmp.getPixels();

    int size[3];
    imIn.getSize(size);
    size_t offset;
    int x, y, z;
    T2 infinite = ImDtTypes<T2>::max();
    long int min;

    // H(x,u) is a minimizer, = MIN(h: 0 <= h < u & Any (i: 0 <= i < u : f(x,h)
    // <= f(x,i)) : h )
    long int size_array = MAX(size[0], size[1]);
    vector<long int> s(size_array); // sets of the least minimizers that occurs
                                    // during the scan from left to right.
    vector<long int> t(
        size_array); // sets of points with the same least minimizer
    s[0]       = 0;
    t[0]       = 0;
    long int q = 0;
    long int w;

    long int tmpdist, tmpdist2;

    for (z = 0; z < size[2]; ++z) {
#ifdef USE_OPEN_MP
#pragma omp for private(offset, x, y, min)
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
            pixelsTmp[offset + y * size[0]] =
                (1 + pixelsTmp[offset + (y - 1) * size[0]] > infinite)
                    ? infinite
                    : 1 + pixelsTmp[offset + (y - 1) * size[0]];
          }
        }
        // SCAN 2
        for (y = size[1] - 2; y >= 0; --y) {
          min = (pixelsTmp[offset + (y + 1) * size[0]] + 1 > infinite)
                    ? infinite
                    : pixelsTmp[offset + (y + 1) * size[0]] + 1;
          if (min < pixelsTmp[offset + y * size[0]])
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
        for (int u = 1; u < size[0]; ++u) {
          tmpdist = (t[q] > s[q]) ? t[q] - s[q] : s[q] - t[q];
          tmpdist = (tmpdist >= pixelsTmp[offset + s[q]])
                        ? tmpdist
                        : pixelsTmp[offset + s[q]];
          tmpdist2 = (t[q] > u) ? t[q] - u : u - t[q];
          tmpdist2 = (tmpdist2 >= pixelsTmp[offset + u])
                         ? tmpdist2
                         : pixelsTmp[offset + u];

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
          if (q < 0) {
            q    = 0;
            s[0] = u;
          } else {
            if (pixelsTmp[offset + s[q]] <= pixelsTmp[offset + u]) {
              w = (s[q] + pixelsTmp[offset + u] >= (s[q] + u) / 2)
                      ? s[q] + pixelsTmp[offset + u]
                      : (s[q] + u) / 2;
            } else {
              w = (u - pixelsTmp[offset + s[q]] >= (s[q] + u) / 2)
                      ? (s[q] + u) / 2
                      : u - pixelsTmp[offset + s[q]];
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
        for (int u = size[0] - 1; u >= 0; --u) {
          pixelsOut[offset + u] = (u > s[q]) ? u - s[q] : s[q] - u;
          pixelsOut[offset + u] =
              (pixelsOut[offset + u] >= pixelsTmp[offset + s[q]])
                  ? pixelsOut[offset + u]
                  : pixelsTmp[offset + s[q]];
          if (u == t[q])
            q--;
        }
      }
    }
    return RES_OK;
  }

  /*
   * Euclidean Distance function.
   */
  template <class T1, class T2>
  RES_T distanceEuclidean(const Image<T1> &imIn, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);
    Image<T2> tmp(imIn);

    typedef Image<T1> imageInType;
    typedef typename imageInType::lineType lineInType;
    typedef Image<T2> imageOutType;
    typedef typename imageOutType::lineType lineOutType;

    lineInType pixelsIn   = imIn.getPixels();
    lineOutType pixelsOut = imOut.getPixels();
    lineOutType pixelsTmp = tmp.getPixels();

    int size[3];
    imIn.getSize(size);
    size_t nbrPixelsPerSlice = size[0] * size[1];
    size_t offset;
    int x, y, z;
    T2 infinite = ImDtTypes<T2>::max();
    // JOE long int min;

    // H(x,u) is a minimizer, = MIN(h: 0 <= h < u & Any (i: 0 <= i < u : f(x,h)
    // <= f(x,i)) : h )
    vector<long int> s(size[0]); // sets of the least minimizers that occurs
                                 // during the scan from left to right.
    vector<long int> t(size[0]); // sets of points with the same least minimizer
    long int q = 0;
    long int w;

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

    return RES_OK;
  }

  /*
   * Geodesic Distance Function
   */
  template <class T1, class T2>
  RES_T distanceGeodesic(const Image<T1> &imIn, const Image<T1> &imMask,
                         Image<T2> &imOut, const StrElt &se)
  {
    ASSERT_ALLOCATED(&imIn, &imOut, &imMask);
    ASSERT_SAME_SIZE(&imIn, &imOut, &imMask);

    ImageFreezer freeze(imOut);

    Image<UINT32> imOffset(imIn);

    typedef Image<T1> imageInType;
    typedef typename imageInType::lineType lineInType;
    typedef Image<T2> imageOutType;
    typedef typename imageOutType::lineType lineOutType;

    lineInType pixelsIn, pixelsMask;
    lineOutType pixelsOut = imOut.getPixels();

    Image<T1> tmp(imIn);

    vector<IntPoint> sePts;
    UINT sePtsNbr;

    typedef size_t OffsetT;
    HierarchicalQueue<T1, OffsetT> hq;
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

    size_t x0, y0, z0;
    size_t x1, y1, z1;
    size_t x2, y2, z2;

    // off_t p0, p1, p2, p3, oo;
    off_t p0, p1, p2;

    float current_dist, wd;

    size_t imSize[3];
    imIn.getSize(imSize);
    size_t maxOffset = imSize[2] * imSize[1] * imSize[0];
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
    for (off_t i = 0; i < (off_t) maxOffset; ++i) {
      if ((pixelsIn[i] > T1(0)) && (pixelsMask[i] > 0)) {
        imIn.getCoordsFromOffset(i, x0, y0, z0);
        //	  std::cout<<"PROCESS:"<<x0<<","<<y0<<"\n";
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
              //		std::cout<<"INIT push:"<<x0<<","<<y0<<"\n";
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

      for (off_t i = 0; i < sePtsNbr; i++) {
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

            pixelsOut[p1] = current_dist;

          } // if wk[p_suiv] != DONE
        }   // x1 no border
      }     // for each ngb of p
    }       // while § EMPTY
    return RES_OK;
  } // END distanceGeodesic

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

  /** @} */
  /** @endcond */
} // namespace smil

#endif // _D_MORPHO_DISTANCE_HPP
