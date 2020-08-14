/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2020, Centre de Morphologie Mathematique
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
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   A filter to convert range of pixel values
 *
 * History :
 *   - 08/03/2019 - by Jose-Marcio Martins da Cruz
 *     Just created it
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_SCALE_FILTER_H_
#define _D_SCALE_FILTER_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   AddonFilters
   * @defgroup  AddonScaleRangeFilter      Image Scale Range of Values
   *
   * @brief Filters to map a range of values in a input image into a different
   * range of values in the output image.
   *
   * @author Jose-Marcio Martins da Cruz
   * @{ */

  /**
   * @brief imageScaleRange : Linear conversion of pixels values to the range
   * [Min, Max]
   *
   * Values in the input image are linearly mapped into the output image with
   * the following rules :
   * - if <b>imIn(x) <= inMin</b>, imOut(x) will be mapped in the range <b>[0,
   * outMin]</b>
   * - if <b>inMin < imIn(x) <= inMax</b>, imOut(x) will be mapped in the range
   * <b>[outMin, outMax]</b>
   * - if <b>imIn(x) > inMax</b>, imOut(x) will be mapped in the range
   * <b>[outMax, max(T2)]</b>
   *
   * @param[in] imIn : input Image
   * @param[in] inMin, inMax : control range in the input image
   * @param[in] outMin, outMax : control range in the output image
   * @param[out] imOut : output Image
   *
   * @note Not Yet Implemented
   */
  template <class T1, class T2>
  RES_T imageScaleRange(const Image<T1> &imIn, const T1 inMin, const T1 inMax,
                        const T2 outMin, const T2 outMax, Image<T2> &imOut);

  /**
   * @brief imageScaleRange : Linear conversion of pixel values to the range
   * [Min, Max]
   *
   * Maps a range of values in the input image into the range <b>[Min, Max]</b>
   * in the output image.
   *
   * If @b onlyNonZero is @b true uses <b>[minVal(), maxVal()]</b> as the range
   * of values in the input image, otherwise, uses the full range of values.
   *
   * @param[in] imIn : input image
   * @param[in] Min : Minimum value in the output image
   * @param[in] Max : Maximum value in the output image
   * @param[out] imOut : output Image
   * @param[in] onlyNonZero : defines how to find input image range of values
   */
  template <class T1, class T2>
  RES_T imageScaleRange(const Image<T1> &imIn, const T2 Min, const T2 Max,
                        Image<T2> &imOut, bool onlyNonZero = true);

  /**
   * @brief imageScaleRange : Linear conversion of pixels values to the
   * domain range
   *
   * Maps a range in the input image into the  range <b>[min(T2), max(T2)]</b>
   * in the output image.
   *
   * If @b onlyNonZero is @b true uses <b>[minVal(), maxVal()]</b> as the range
   * of values in the input image, otherwise, uses the full range of values.
   *
   * @param[in] imIn : input Image
   * @param[out] imOut : output Image
   * @param[in] onlyNonZero : defines how to find input image range of values
   */
  template <class T1, class T2>
  RES_T imageScaleRange(const Image<T1> &imIn, Image<T2> &imOut,
                        bool onlyNonZero = true);

  /**
   * @brief imageScaleRangeSCurve : S Curve transform
   *
   * This function emulates the <b>S Curve</b> caracteristic of film
   * photography.
   *
   * Use a sigmoid function centered at @b pivot with derivative @b ratio.
   *
   * One use of this filter is to set (increase or decrease) the contrast in the
   * neighborhood of the @b pivot.
   *
   * @param[in] imIn : input Image
   * @param[in] pivot :
   * * if 0, takes the median of the histogram of input image as pivot
   * * otherwise, use this value
   * @param[in] ratio : derivative of output image at pivot value
   * @param[out] imOut : output Image
   */
  template <class T1, class T2>
  RES_T imageScaleRangeSCurve(const Image<T1> &imIn, const T1 pivot,
                              const double ratio, Image<T2> &imOut);

  /** @} */
} // namespace smil

#include "private/filterRangeScale/filterRangeScale.hpp"

#endif // _D_SCALE_FILTER_H_
