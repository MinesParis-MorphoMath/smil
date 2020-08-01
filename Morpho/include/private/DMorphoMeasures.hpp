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

#ifndef _D_MORPHO_MEASURES_HPP
#define _D_MORPHO_MEASURES_HPP

#include "Core/include/DImage.h"
#include "DMorphoBase.hpp"

namespace smil
{
  /**
   * @ingroup Morpho
   * @defgroup MorphoMeasures Measures
   * @{
   */

  /**
   * Granulometry by openings.
   *
   * Performs openings of increasing size (using steps of @b stepSize) and
   * measure the corresponding volume difference.
   *
   * If @b CDF is true, return a Cumulative Distribution Function
   */
  template <class T>
  vector<double>
  measGranulometry(const Image<T> &imIn, const StrElt &se = DEFAULT_SE,
                   const unsigned int stepSize = 1, bool CDF = true,
                   const unsigned int maxSeSize = 0)
  {
    vector<double> res;

    ASSERT(imIn.isAllocated(), res);

    Image<T> imEro(imIn, true); // clone
    Image<T> imOpen(imIn);

    size_t seSize = stepSize;

    double v0 = vol(imIn);
    double v1;
    T minv = minVal(imEro);

    do {
      erode(imEro, imEro, se(stepSize));
      dilate(imEro, imOpen, se(seSize));
      v1 = vol(imOpen);
      res.push_back(v0 - v1);
      v0 = v1;
      seSize += stepSize;
    } while (maxVal(imEro) > minv || (maxSeSize > 0 && maxSeSize > seSize));

    if (CDF) {
      double aSum = 0;
      for (size_t i = 0; i < res.size(); i++)
        aSum += res[i];
      res[0] /= aSum;
      for (size_t i = 1; i < res.size(); i++)
        res[i] = res[i] / aSum + res[i - 1];
    }

    return res;
  }

  /** @}*/

} // namespace smil

#endif // _D_MORPHO_MEASURES_HPP
