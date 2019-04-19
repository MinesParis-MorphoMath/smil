/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2019, Centre de Morphologie Mathematique
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
 *   2D Gabor filter implementation by Vincent Morard
 *
 * History :
 *   - 08/03/2019 - by Jose-Marcio Martins da Cruz
 *     Just created it
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_NORMALIZE_FILTER_H_
#define _D_NORMALIZE_FILTER_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   AddonFilters
   * @defgroup  AddonScaleFilter      Image Normalize
   *
   * @brief Various filters to normalize images
   *
   * @author Jose-Marcio Martins da Cruz
   * @{ */

  /**
   * @brief ImNormalize : Linear conversion of pixels values to the range [Min,
   * Max]
   * @param[in] imIn : input Image
   * @param[in] Min : Minimum value in the output image
   * @param[in] Max : Maximum value in the output image
   * @param[out] imOut : output Image
   */
  template <class T1, class T2>
  RES_T ImNormalize(const Image<T1> &imIn, const T2 Min, const T2 Max,
                    Image<T2> &imOut);

  /**
   * @brief ImNormalizeAuto : Linear conversion of pixels values to the domain
   * range
   * @param[in] imIn : input Image
   * @param[out] imOut : output Image
   */
  template <class T1, class T2>
  RES_T ImNormalizeAuto(const Image<T1> &imIn, Image<T2> &imOut);

  /**
   * @brief ImNormalizeSCurve : S Curve transform
   *
   * This function emulates the "S Curve" caracteristic of film photography.
   *
   * Use a sigmoid function centered at "pivot" with derivative "ratio". 
   * 
   * @param[in] imIn : input Image
   * @param[in] pivot : 
   * * if 0, takes the median of the histogram of input image as pivot
   * * otherwise, use this value
   * @param[in] ratio : derivative of output image at pivot value
   * @param[out] imOut : output Image
   */
  template <class T1, class T2>
  RES_T ImNormalizeSCurve(const Image<T1> &imIn, const T1 pivot,
                        const double ratio, Image<T2> &imOut);

  /** @} */
} // namespace smil

#include "private/filterNormalize/filterNormalize.hpp"

#endif // _D_NORMALIZE_FILTER_H_
