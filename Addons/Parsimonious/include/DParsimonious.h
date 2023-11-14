/*
 * __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2023, Centre de Morphologie Mathematique
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
 *   Parsimonious incomplete-path openings
 *
 * History :
 *   - 08/06/2020 - by Jose-Marcio Martins da Cruz
 *     Porting from Petr Dokladal code in Morph-M
 *
 * __HEAD__ - Stop here !
 */

#ifndef __FAST_PIPATH_OPENING_H__
#define __FAST_PIPATH_OPENING_H__

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup AddonParsimonious Parsimonious Path Opening
   * @{
   *
   *   @brief Parsimonious incomplete-path openings
   *
   *   @details
   *   Implements Parsimonious Incomplete-Path Openings using the Rank Opening
   *   (See Pierre Soille @cite Soille_2002 and 
   *   Petr Dokladal \cite dokladal:hal-01478089).
   *
   *   The rank opening is implemented as a rank filter followed by a dilation.
   *   It uses histograms.
   *
   *   @author Petr Dokladal (based on Vincent Morard code),
   *   @author Jose-Marcio Martins da Cruz (port to Smil)
   *
   * @par Example of use
   *
   * @include demo-parsimonious-path-opening.py
   *
   */

  /**
   * @brief parsimoniousPathOpening
   * 
   * @param[in]  imIn : input Image
   * @param[in]  Size : size of Structuring Element
   * @param[in]  tolerance :
   * @param[in]  step :
   * @param[in]  rebuild : do Geodesic Rebuild after PathOpening
   * @param[out] imOut : output Image
   *
   * @warning This function may be very slow on UINT32 image type.
   *
   */
  template <class T>
  RES_T parsimoniousPathOpening(Image<T> &imIn, int Size,
                                            int tolerance, int step,
                                            bool rebuild, Image<T> &imOut);

  /**
   * @brief parsimoniousPathOpening
   * 
   * @param[in]  imIn : input Image
   * @param[in]  Size : size of Structuring Element
   * @param[in]  tolerance :
   * @param[in]  step :
   * @param[out] imOut : output Image
   */
  template <class T>
  RES_T parsimoniousPathOpening(Image<T> &imIn, int Size,
                                            int tolerance, int step,
                                            Image<T> &imOut)
  {
    return parsimoniousPathOpening(imIn, Size,
                                               tolerance, step,
                                               true, imOut);
  }

  /** @} */

} // namespace smil

/** @cond */
#include "private/FastIncompletePathOpening.hpp"
/** @endcond */
#endif // __FAST_PIPATH_OPENING_H__
