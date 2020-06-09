/*
 * __HEAD__
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

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup Parsimonious Parsimonious Path Opening
   * @{
   *
   *   Parsimonious incomplete-path openings
   *   Author : Petr Dokladal (based on V. Morard's code)
   *   implements Parsimonious Incomplete-Path Openings using the Rank Opening
   *   (see Pierre Soille, On morphological operators based on rank filters,
   *    Pattern Recognition 35 (2002) 527-535)
   *   The rank opening is implemented as a rank filter followed by a dilation.
   *   It uses histograms.
   *
   * @author Petr Dokladal / Jose-Marcio Martins da Cruz
   *
   * @{ */

  /**
   * @brief ImNormalizeAuto : Linear conversion of pixels values to the domain
   * range
   * @param[in]  imIn : input Image
   * @param[in]  Size :
   * @param[in]  tolerance :
   * @param[in]  step :
   * @param[in]  rec :
   * @param[out] imOut : output Image
   */
  template <class T>
  RES_T ImParsimoniousIncompletePathOpening(Image<T> &imIn, int Size,
                                            int tolerance, int step,
                                            bool rebuild, Image<T> &imOut);

  /** @} */

} // namespace smil

#include "private/private/DParsimonious.hpp"

#endif __FAST_PIPATH_OPENING_H__