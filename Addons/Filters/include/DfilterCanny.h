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

#ifndef _D_FILTER_CANNY_H_
#define _D_FILTER_CANNY_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   AddonFilters
   * @defgroup  AddonFilterCanny        Canny Filter (2D)
   * @brief A 2D Canny filter implementation by Vincent Morard
   *
   * Canny edge detection: Canny's aim was to discover the optimal edge 
   * detection algorithm. In this situation, an "optimal" edge detector means:
   *
   * - good detection - the algorithm should mark as many real edges in the 
   * image as possible.
   *
   * - good localization - edges marked should be as close as possible to the 
   * edge in the real image.
   *
   * - minimal response - a given edge in the image should only be marked once,
   * and where possible, image noise should not create false edges.
   *
   * To satisfy these requirements Canny used the calculus of variations - a 
   * technique which finds the function which optimizes a given functional.
   *
   * The optimal function in Canny's detector is described by the sum of four 
   * exponential terms, but can be approximated by the first derivative of a 
   * Gaussian.
   *
   * @see
   * * Canny Edge detector <a href="https://en.wikipedia.org/wiki/Canny_edge_detector">
   *    on Wikipedia</a>
   * * John Canny, A computational approach to edge detection, IEEE Pami, 
   * vol. 8, n° 6, novembre 1986, pp 679-698
   *
   * @author Vincent Morard / Jose-Marcio Martins da Cruz
   * @{ */

  /**
   * @brief ImCannyEdgeDetection Canny Filter
   * @param[in] imIn : input Image
   * @param[in] sigma :
   * @param[out] imOut : output Image
   */
  template <class T1, class T2>
  RES_T ImCannyEdgeDetection(const Image<T1> &imIn, const double sigma,
                             Image<T2> &imOut);

  /** @} */
} // namespace smil

#include "private/filterCanny/filterCanny.hpp"

#endif // _D_FILTER_CANNY_H_
