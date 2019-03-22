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

#ifndef _DAPPLYTHRESHOLD_HPP_
#define _DAPPLYTHRESHOLD_HPP_

#include "Morpho/include/DMorpho.h"
#include "DUtils.h"

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup AddonThresh 
   * @{
   */

  /**
   * Apply Threshold
   *
   */
  template <class T>
  RES_T applyThreshold(const Image<T> &_im_, const vector<T> &modes,
                       Image<T> &_out_)
  {
    size_t S[3];
    _im_.getSize(S);

    size_t s = S[0] * S[1] * S[2];

    T *out = _out_.getPixels();
    T *im  = _im_.getPixels();

    UINT SMIL_UNUSED nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
    for (size_t p = 0; p < s; ++p) {
      out[p] = modes[im[p]];
    }
    return RES_OK;
  }

  /**
   * Area Threshold
   *
   */
  template <class T>
  RES_T areaThreshold(const Image<T> &_im_, const T &threshold, Image<T> &_out_)
  {
    map<T, double> m = measAreas(_im_, false);
    vector<T> v(maxVal<T>(_im_) + 1, 0);
    for (typename map<T, double>::iterator it = m.begin(); it != m.end();
         ++it) {
      if (it->second > threshold)
        v[it->first] = it->first;
      else
        v[it->first] = 0;
    }
    applyThreshold(_im_, v, _out_);
    return RES_OK;
  }

  /**
   * Range Threshold
   *
   */
  template <class T>
  RES_T rangeThreshold(const Image<T> &_im_, const T &threshold,
                       Image<T> &_out_)
  {
    vector<T> v(maxVal<T>(_im_) + 1, 0);

    size_t S[3];
    _im_.getSize(S);
    size_t s = S[0] * S[1] * S[2];

    Image<T> _tmp_ = Image<T>(_im_);
    dist_cross_3d_per_label(_im_, _tmp_);

    T *im  = _im_.getPixels();
    T *tmp = _tmp_.getPixels();

    UINT SMIL_UNUSED nthreads = Core::getInstance()->getNumberOfThreads();
    //        #pragma omp parallel for num_threads(nthreads)
    for (size_t p = 0; p < s; ++p) {
      v[im[p]] = (v[im[p]] < tmp[p]) ? tmp[p] : v[im[p]];
    }
    applyThreshold(_im_, v, _tmp_);
    compare(_tmp_, ">", threshold, _im_, T(0), _out_);

    return RES_OK;
  }

  template <class T1, class T2>
  RES_T dist_cross_3d_per_label(const Image<T1> &imIn, Image<T2> &imOut)
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

    int size[3];
    imIn.getSize(size);
    size_t offset;
    int x, y, z;
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
          } else if (pixelsIn[offset + y * size[0]] !=
                     pixelsIn[offset + (y - 1) * size[0]]) {
            pixelsOut[offset + y * size[0]] = T2(1);
          } else {
            pixelsOut[offset + y * size[0]] =
                (1 + pixelsOut[offset + (y - 1) * size[0]] > infinite)
                    ? infinite
                    : 1 + pixelsOut[offset + (y - 1) * size[0]];
          }
        }

        for (y = size[1] - 2; y >= 0; --y) {
          if (pixelsIn[offset + (y + 1) * size[0]] !=
              pixelsIn[offset + y * size[0]]) {
            min = 1;
          } else {
            min = (pixelsOut[offset + (y + 1) * size[0]] + 1 > infinite)
                      ? infinite
                      : pixelsOut[offset + (y + 1) * size[0]] + 1;
          }
          if (min < pixelsOut[offset + y * size[0]])
            pixelsOut[offset + y * size[0]] = min;
        }
      }

#ifdef USE_OPEN_MP
#pragma omp for private(x, y, offset)
#endif // USE_OPEN_MP
      for (y = 0; y < size[1]; ++y) {
        offset = z * size[1] * size[0] + y * size[0];
        for (x = 1; x < size[0]; ++x) {
          if (pixelsOut[offset + x] != 0) {
            if (pixelsIn[offset + x] != pixelsIn[offset + x - 1]) {
              pixelsOut[offset + x] = T2(1);
            } else if (pixelsOut[offset + x] > pixelsOut[offset + x - 1]) {
              pixelsOut[offset + x] = pixelsOut[offset + x - 1] + 1;
            }
          }
        }
        for (x = size[0] - 2; x >= 0; --x) {
          if (pixelsOut[offset + x] != 0) {
            if (pixelsIn[offset + x] != pixelsIn[offset + x + 1]) {
              pixelsOut[offset + x] = T2(1);
            } else if (pixelsOut[offset + x] > pixelsOut[offset + x + 1]) {
              pixelsOut[offset + x] = pixelsOut[offset + x + 1] + 1;
            }
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
          if (pixelsOut[offset + z * size[1] * size[0]] != 0) {
            if (pixelsIn[offset + z * size[1] * size[0]] !=
                pixelsIn[offset + (z - 1) * size[1] * size[0]]) {
              pixelsOut[offset + z * size[1] * size[0]] = T2(1);
            } else if (pixelsOut[offset + z * size[1] * size[0]] >
                       pixelsOut[offset + (z - 1) * size[1] * size[0]]) {
              pixelsOut[offset + z * size[1] * size[0]] =
                  pixelsOut[offset + (z - 1) * size[1] * size[0]] + 1;
            }
          }
        }
        for (z = size[2] - 2; z >= 0; --z) {
          if (pixelsOut[offset + z * size[1] * size[0]] != 0) {
            if (pixelsIn[offset + z * size[1] * size[0]] !=
                pixelsIn[offset + (z + 1) * size[1] * size[0]]) {
              pixelsOut[offset + z * size[1] * size[0]] = T2(1);
            } else if (pixelsOut[offset + z * size[1] * size[0]] >
                       pixelsOut[offset + (z + 1) * size[1] * size[0]]) {
              pixelsOut[offset + z * size[1] * size[0]] =
                  pixelsOut[offset + (z + 1) * size[1] * size[0]] + 1;
            }
          }
        }
      }
    }

    return RES_OK;
  }

  template <class T1, class T2>
  RES_T dist_per_label(const Image<T1> &imIn, Image<T2> &imOut,
                       const StrElt &se = DEFAULT_SE)
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

    fill<T2>(imOut, 0);

    Image<T1> tmp(imIn);
    Image<T1> tmp2(imIn);
    Image<T1> tmp3(imIn);

    ASSERT(compare(imIn, ">", T1(0), T1(1), T1(0), tmp) == RES_OK);
    ASSERT(erode(tmp, tmp2, Cross3DSE()) == RES_OK);
    ASSERT(sub(tmp, tmp2, tmp3) == RES_OK);
    ASSERT(copy(tmp3, imOut) == RES_OK);

    queue<size_t> *level      = new queue<size_t>();
    queue<size_t> *next_level = new queue<size_t>();
    queue<size_t> *swap;

    T2 cur_level = T2(2);

    int size[3];
    imIn.getSize(size);
    int64_t s = size[0] * size[1] * size[2];

    size_t cur;
    int64_t x, y, z, n_x, n_y, n_z;

    vector<IntPoint> sePoints = se.points;
    vector<IntPoint>::iterator pt;

    bool oddLine;

    for (int64_t p = 0; p < s; ++p) {
      if (pixelsOut[p] > 0) {
        level->push(p);
      }
    }

    do {
      while (!level->empty()) {
        cur = level->front();
        pt  = sePoints.begin();
        z   = cur / (size[1] * size[0]);
        y   = (cur - z * size[1] * size[0]) / size[0];
        x   = cur - y * size[0] - z * size[0] * size[1];

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
              pixelsIn[n_x + (n_y) *size[0] + (n_z) *size[1] * size[0]] ==
                  pixelsIn[x + y * size[0] + z * size[1] * size[0]]) {
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

    return RES_OK;
  }

  /**
   * rasterLabels
   *
   */
  template <class T> RES_T rasterLabels(const Image<T> &_im_, Image<T> &_out_)
  {
    size_t S[3];
    _im_.getSize(S);

    size_t s = S[0] * S[1] * S[2];

    T *out = _out_.getPixels();
    T *im  = _im_.getPixels();
    map<T, T> m;

    T count = 1;

    // UINT nthreads = Core::getInstance()->getNumberOfThreads ();
    //#pragma omp parallel for num_threads(nthreads)
    for (size_t p = 0; p < s; ++p) {
      if (im[p] != 0) {
        typename map<T, T>::iterator it = m.find(im[p]);
        if (it == m.end())
          m[im[p]] = count++;
        out[p] = it->second;
      }
    }
    return RES_OK;
  }

  /**
   *  Find Triple Points
   *
   */
  template <class T1, class T2>
  RES_T findTriplePoints(const Image<T1> &_im_, const Image<T2> &_skiz_,
                         Image<T2> &_out_, const UINT &val, const StrElt &_se_)
  {
    T1 *in   = _im_.getPixels();
    T2 *skiz = _skiz_.getPixels();
    T2 *out  = _out_.getPixels();
    fill<T2>(_out_, T2(0));

    size_t S[3];
    _im_.getSize(S);
    size_t nbrPixelsInSlice = S[0] * S[1];
    size_t nbrPixels        = nbrPixelsInSlice * S[2];
    StrElt se               = _se_;
    UINT sePtsNumber        = se.points.size();

    UINT nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel num_threads(nthreads)
    {
      index p, q, l;
      UINT pts;
      map<T1, bool> m;

#pragma omp for
      ForEachPixel(p)
      {
        ForEachNeighborOf(p, q)
        {
          m[in[q.o]] = true;
        }
        ENDForEachNeighborOf if (m.size() >= val && skiz[p.o] != 0 &&
                                 skiz[p.o] != ImDtTypes<T2>::max())
        {
          out[p.o] = ImDtTypes<T2>::max();
        }
        m.clear();
      }
      ENDForEachPixel
    }
    return RES_OK;
  }

  /**
   * Extend Triple Points
   *
   */
  template <class T>
  RES_T extendTriplePoints(Image<T> &_triple_, const Image<T> &_skiz_,
                           const StrElt &_se_)
  {
    T *triple = _triple_.getPixels();
    T *skiz   = _skiz_.getPixels();

    size_t S[3];
    _triple_.getSize(S);
    size_t nbrPixelsInSlice = S[0] * S[1];
    size_t nbrPixels        = nbrPixelsInSlice * S[2];
    StrElt se               = _se_.noCenter();
    UINT sePtsNumber        = se.points.size();

    UINT nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel num_threads(nthreads)
    {
      index p, q;
      UINT pts;
      queue<size_t> c;
#pragma omp for
      ForEachPixel(p)
      {
        if (skiz[p.o] == ImDtTypes<T>::max() || skiz[p.o] == 0) {
          triple[p.o] = 0;
        }
        if (triple[p.o] == ImDtTypes<T>::max()) {
          c.push(p.o);
        }
      }
      ENDForEachPixel while (!c.empty())
      {
        p.o = c.front();
        c.pop();
        IndexToCoor(p);

        ForEachNeighborOf(p, q)
        {
          if (skiz[q.o] == skiz[p.o] && triple[q.o] != ImDtTypes<T>::max()) {
            c.push(q.o);
            triple[p.o] = ImDtTypes<T>::max();
            triple[q.o] = ImDtTypes<T>::max();
          }
        }
        ENDForEachNeighborOf
      }
    }

    return RES_OK;
  }

  /**
   * pruneSKIZ
   *
   */
  template <class T>
  RES_T pruneSKIZ(const Image<T> &_im_, Image<T> &_out_, const StrElt &_se_)
  {
    T *in  = _im_.getPixels();
    T *out = _out_.getPixels();

    fill<T>(_out_, T(0));

    size_t S[3];
    _im_.getSize(S);
    size_t nbrPixelsInSlice = S[0] * S[1];
    size_t nbrPixels        = nbrPixelsInSlice * S[2];
    StrElt se               = _se_;
    UINT sePtsNumber        = se.points.size();

    UINT nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel num_threads(nthreads)
    {
      index p, q;
      UINT pts;
      bool up, down;

#pragma omp for
      ForEachPixel(p)
      {
        up   = false;
        down = false;
        if (in[p.o] > 0 && in[p.o] != ImDtTypes<T>::max()) {
          IndexToCoor(p);
          ForEachNeighborOf(p, q)
          {
            if (in[q.o] != ImDtTypes<T>::max()) {
              if (in[q.o] >= in[p.o] + 1) {
                up = true;
              }
              if (in[q.o] <= in[p.o] - 1) {
                down = true;
              }
            }
          }
          ENDForEachNeighborOf if (!up || !down)
          {
            out[p.o] = in[p.o];
          }
        }
      }
      ENDForEachPixel
    }

    /*
            UINT nthreads = Core::getInstance()->getNumberOfThreads ();
    //        #pragma omp parallel num_threads(nthreads)
            {
                index p, q;
                UINT pts;
                queue <size_t> c;

                #pragma omp for
                ForEachPixel (p)
                {
                    if (triple[p.o])
                    {
                        c.push (p.o);
                    }
                }
                ENDForEachPixel

                while (!c.empty())
                {
                    p.o = c.front();
                    c.pop();
                    IndexToCoor (p);

                    ForEachNeighborOf (p,q)
                    {
                        if (in[p.o] >= in[q.o]+1 && in[q.o] != 0)
                        {
                            out[p.o] = ImDtTypes<T>::max();
                            c.push (q.o);
                        }
                    }
                    ENDForEachNeighborOf
                }
            }
    */

    return RES_OK;
  }
  /** @} */
} // namespace smil

#endif // _DAPPLYTHRESHOLD_HPP_
