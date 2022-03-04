/*
 * __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2022, Centre de Morphologie Mathematique
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
 *   This file does... some very complex morphological operation...
 *
 * History :
 *   - XX/XX/XXXX - by Joe.Denver
 *     Just created it...
 *   - XX/XX/XXXX - by Joe.Denver
 *     Modified something
 *
 * __HEAD__ - Stop here !
 */


#ifndef _DSTOCHASTICWS_H_
#define _DSTOCHASTICWS_H_

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup AddonStochasticWatershed Stochastic Watershed
   * @{
   *
   * Put some brief (or long) description here
   *
   * @author Theodore Chabardes
   */

  /**
   * stochasticWatershedParallel
   *
   * @param[in] primary
   * @param[in] gradient
   * @param[out] out
   * @param[in] n_seeds
   * @param[in] se
   */
  template <class labelT, class T>
  void stochasticWatershedParallel(const Image<labelT> &primary,
                                   const Image<T> &gradient, Image<labelT> &out,
                                   const size_t &n_seeds, const StrElt &se);

  /**
   * stochasticWatershed
   *
   * @param[in] primary
   * @param[in] gradient
   * @param[out] out
   * @param[in] n_seeds
   * @param[in] se
   */
  template <class labelT, class T>
  void stochasticWatershed(const Image<labelT> &primary,
                           const Image<T> &gradient, Image<labelT> &out,
                           const size_t &n_seeds, const StrElt &se);
                           
  /**
   * stochasticFlatZonesParallel
   *
   * @param[in] primary
   * @param[in] gradient
   * @param[out] out
   * @param[in] n_seeds
   * @param[in] t0
   * @param[in] se
   */
  template <class labelT, class T>
  size_t stochasticFlatZonesParallel(const Image<labelT> &primary,
                                     const Image<T> &gradient,
                                     Image<labelT> &out, const size_t &n_seeds,
                                     const double &t0, const StrElt &se);
                                     
  /**
   *  Over Segmentation Correction
   *
   * @param[in] primary
   * @param[in] gradient
   * @param[out] out
   * @param[in] n_seeds
   * @param[in] t0
   * @param[in] se
   */
  template <class labelT, class T>
  size_t stochasticFlatZones(const Image<labelT> &primary,
                             const Image<T> &gradient, Image<labelT> &out,
                             const size_t &n_seeds, const double &t0,
                             const StrElt &se);

  /**
   *  Over Segmentation Correction
   *
   * @param[in] primary
   * @param[in] gradient
   * @param[out] out
   * @param[in] n_seeds
   * @param[in] r0
   * @param[in] se
   */
  template <class labelT, class T>
  size_t overSegmentationCorrection(const Image<labelT> &primary,
                                    const Image<T> &gradient,
                                    Image<labelT> &out, const size_t &n_seeds,
                                    const double &r0, const StrElt &se);

/** @} */

} // namespace smil

/** @cond */
#include "private/DStochasticWatershed.hpp"
/** @endcond */

#endif // _DSTOCHASTICWS_H_

