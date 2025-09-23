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

#ifndef _D_COLOR_MORPHO_H
#define _D_COLOR_MORPHO_H

#include "Core/include/private/DImage.hxx"
#include "Core/include/DColor.h"
#include "Morpho/include/DMorpho.h"

#include "NSTypes/RGB/include/DRGB.h"

namespace smil
{
  /**
   * @addtogroup AddonColor
   * @{
   */

  /**
   * @brief gradientLAB
   * @param[in] imIn : input RGB Image
   * @param[out] imOut : output Image (UINT8 Gray Level)
   * @param[in] se : Structuring Element
   * @param[in] convertFirstToLAB : Convert to LAB
   */
  RES_T gradientLAB(const Image<RGB> &imIn, Image<UINT8> &imOut,
                    const StrElt &se                = DEFAULT_SE,
                    bool          convertFirstToLAB = true);

  /**
   * @brief gradientLAB
   * @param[in] imIn : input RGB Image
   * @param[in] se : Structuring Element
   * @param[in] convertFirstToLAB : Convert to LAB
   * @return Image (UINT8 Gray Level)
   */
  Image<UINT8> gradientLAB(const Image<RGB> &imIn,
                           const StrElt     &se                = DEFAULT_SE,
                           bool              convertFirstToLAB = true);

  /**
   * @brief gradientHLS
   * @param[in] imIn : input RGB Image
   * @param[out] imOut : output Image (UINT8 Gray Level)
   * @param[in] se : Structuring Element
   * @param[in] convertFirstToHLS : Convert to LAB
   */
  RES_T gradientHLS(const Image<RGB> &imIn, Image<UINT8> &imOut,
                    const StrElt &se                = DEFAULT_SE,
                    bool          convertFirstToHLS = true);

  /**
   * @brief gradientHLS
   * @param[in] imIn : input RGB Image
   * @param[in] se : Structuring Element
   * @param[in] convertFirstToHLS : Convert to HLS
   * @return Image  (UINT8 Gray Level)
   */
  Image<UINT8> gradientHLS(const Image<RGB> &imIn,
                           const StrElt     &se                = DEFAULT_SE,
                           bool              convertFirstToHLS = true);
  /** @} */
} // namespace smil

#endif // _D_COLOR_MORPHO_H
