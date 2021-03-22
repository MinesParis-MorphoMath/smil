/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2021, Centre de Morphologie Mathematique
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
 *   - XX/XX/XXXX - by Vincent Morard
 *     Just created it
 *   - 21/02/2019 - by Jose-Marcio Martins da Cruz
 *     Formatting and removing some warnings and minor differences
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_SIGMA_FILTER_H_
#define _D_SIGMA_FILTER_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   AddonFilters
   * @defgroup  AddonSigmaFilter        Sigma Filter (2D)
   *
   * @brief A 2D Sigma filter implementation
   *
   * Performs a noise reduction following the Lee paper
   *
   * @see
   *
   * @author Vincent Morard
   * @author Jose-Marcio Martins da Cruz (port from @c Morph-M)
   * @{ */

  /**
   * @brief Performs a noise reduction following the Lee paper
   * @param[in] imIn : input Image
   * @param[in] radius :
   * @param[in] sigma :
   * @param[in] percentageNbMinPixel :
   * @param[in] excludeOutlier :
   * @param[out] imOut : output Image
   */
  template <class T>
  RES_T sigmaFilter(const Image<T> &imIn, const UINT8 radius,
                      const double sigma, const double percentageNbMinPixel,
                      const bool excludeOutlier, Image<T> &imOut);

  /**
   * @brief Performs a noise reduction following the Lee paper (RGB Images)
   * @param[in] imIn : input Image
   * @param[in] radius :
   * @param[in] sigma :
   * @param[in] percentageNbMinPixel :
   * @param[in] excludeOutlier :
   * @param[out] imOut : output Image
   * @warning Yet to be done !!!
   */
  template <class T>
  RES_T sigmaFilterRGB(const Image<T> &imIn, const UINT8 radius,
                         const double sigma, const double percentageNbMinPixel,
                         const bool excludeOutlier, Image<T> &imOut);
  /** @} */
} // namespace smil

#include "private/filterSigma/filterSigma.hpp"

#endif // _D_SIGMA_FILTER_H_

