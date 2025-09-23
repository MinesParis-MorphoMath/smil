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

#ifndef _D_CHABARDES_MEASURES_HPP_
#define _D_CHABARDES_MEASURES_HPP_

#include "Core/include/private/DImage.hpp"
#include "Base/include/private/DBaseMeasureOperations.hpp"
#include "Base/include/private/DLineArith.hpp"

namespace smil
{
  /*
   * @ingroup Addons
   * @addtogroup AddonMeasures
   * @{
   */

  /*
   * Measure Haralick Features
   *
   */
  template <class T>
  vector<double> measHaralickFeatures(Image<T> &imIn, const StrElt &s)
  {
    map<T, UINT> hist = histogram(imIn);
    map<T, T>    equivalence;

    size_t nbr_components = 0;
    for (typename map<T, UINT>::iterator it = hist.begin(); it != hist.end();
         ++it) {
      if (it->second != 0) {
        equivalence.insert(pair<T, T>(it->first, nbr_components));
        nbr_components++;
      }
    }

    size_t S[3];
    imIn.getSize(S);
    size_t nbrPixelsInSlice = S[0] * S[1];
    size_t nbrPixels        = nbrPixelsInSlice * S[2];
    T     *in               = imIn.getPixels();
    StrElt se               = s.noCenter();
    UINT   sePtsNumber      = se.points.size();

    vector<double> vec = vector<double>(nbr_components * nbr_components, 0.0);

    UINT SMIL_UNUSED nthreads = Core::getInstance()->getNumberOfThreads();
    // # pragma omp parallel num_threads(nthreads)
    {
      index_T        p, q;
      UINT           pts;
      vector<double> vec_local =
          vector<double>(nbr_components * nbr_components, 0.0);
      T *counts = new T[nbr_components];
      T  max    = 0;

#pragma omp for
      ForEachPixel(p)
      {
        max = 0;
        for (size_t i = 0; i < nbr_components; ++i) {
          counts[i] = 0;
        }

        ForEachNeighborOf(p, q)
        {
          if (in[q.o] != in[p.o]) {
            counts[equivalence[in[q.o]]]++;
          }
        }
        ENDForEachNeighborOf

            for (uint32_t i = 0; i < nbr_components; ++i)
        {
          max = (counts[i] > counts[max] ||
                 (counts[i] == counts[max] && vec_local[i] < vec_local[max]))
                    ? i
                    : max;
        }

        if (counts[max] != 0)
          vec_local[equivalence[in[p.o]] * nbr_components + max]++;
      }
      ENDForEachPixel

#pragma omp for ordered schedule(static, 1)
          for (int t = 0; t < omp_get_num_threads(); ++t)
      {
#pragma omp ordered
        {
          for (uint64_t i = 0; i < vec.size(); ++i) {
            vec[i] += vec_local[i];
          }
        }
      }
      delete counts;
    }

    return vec;
  }

  /*
   * CrossCorrelation between two phases
   *
   * The direction is given by \b dx, \b dy and \b dz.
   * The lenght corresponds to the max number of steps \b maxSteps
   */
  template <class T>
  vector<double> measCrossCorrelation(const Image<T> &imIn, const T &val1,
                                      const T &val2, size_t dx, size_t dy,
                                      size_t dz, UINT maxSteps = 0,
                                      bool normalize = false)
  {
    vector<double> vec;
    ASSERT(areAllocated(&imIn, NULL), vec);

    size_t s[3];
    imIn.getSize(s);
    if (maxSteps == 0)
      maxSteps = max(max(s[0], s[1]), s[2]) - 1;
    vec.clear();

    typename ImDtTypes<T>::volType   slicesIn = imIn.getSlices();
    typename ImDtTypes<T>::sliceType curSliceIn1;
    typename ImDtTypes<T>::sliceType curSliceIn2;
    typename ImDtTypes<T>::lineType  lineIn1;
    typename ImDtTypes<T>::lineType  lineIn2;
    typename ImDtTypes<T>::lineType  bufLine1 = ImDtTypes<T>::createLine(s[0]);
    typename ImDtTypes<T>::lineType  bufLine2 = ImDtTypes<T>::createLine(s[0]);
    typename ImDtTypes<T>::lineType  val1L    = ImDtTypes<T>::createLine(s[0]);
    fillLine<T>(val1L, s[0], val1);
    typename ImDtTypes<T>::lineType val2L = ImDtTypes<T>::createLine(s[0]);
    fillLine<T>(val2L, s[0], val2);
    equLine<T> eqOp;

    for (UINT len = 0; len <= maxSteps; len++) {
      double prod = 0;
      size_t xLen = s[0] - dx * len;
      size_t yLen = s[1] - dy * len;
      size_t zLen = s[2] - dz * len;

      for (size_t z = 0; z < zLen; z++) {
        curSliceIn1 = slicesIn[z];
        curSliceIn2 = slicesIn[z + len * dz];
        for (UINT y = 0; y < yLen; y++) {
          lineIn1 = curSliceIn1[y];
          lineIn2 = curSliceIn2[y + len * dy];
          eqOp(lineIn1, val1L, xLen, bufLine1);
          eqOp(lineIn2, val2L, xLen, bufLine2);
          for (size_t x = 0; x < xLen; x++) // Vectorized loop
            prod += bufLine1[x] * bufLine2[x];
        }
      }
      if (xLen * yLen * zLen != 0)
        prod /= (xLen * yLen * zLen);
      vec.push_back(prod);
    }

    if (normalize) {
      double orig = vec[0];
      for (vector<double>::iterator it = vec.begin(); it != vec.end(); it++)
        *it /= orig;
    }

    ImDtTypes<T>::deleteLine(bufLine1);
    ImDtTypes<T>::deleteLine(bufLine2);

    return vec;
  }
  /* @}*/
} // namespace smil

#endif // _D_CHABARDES_MEASURES_HPP_
