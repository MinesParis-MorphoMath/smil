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
 *   Gaussian Filters
 *
 * History :
 *   - 12/03/2019 - by Jose-Marcio Martins da Cruz
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_GAUSSIAN_FILTER_H_
#define _D_GAUSSIAN_FILTER_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   AddonFilters
   * @defgroup  AddonGaussianFilter        Gaussian Filter (3D)
   *
   * @brief Gaussian Filter
   *
   *
   * @author Jose-Marcio Martins da Cruz
   * @{ */

  /**
   * @brief filterGaussian Gaussian Filter (3D)
   * @param[in] imIn : input Image
   * @param[in] radius : Gaussian kernel radius
   * @param[out] imOut : output Image
   */
  template <class T>
  RES_T ImGaussianFilter(Image<T> &imIn, int radius, Image<T> &imOut);

  /** @} */
} // namespace smil

#include "private/filterGaussian/gaussianKernel.hpp"

#endif // _D_GAUSSIAN_FILTER_H_
