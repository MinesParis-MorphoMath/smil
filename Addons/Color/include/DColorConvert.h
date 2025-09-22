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

#ifndef _D_COLOR_CONVERT_H
#define _D_COLOR_CONVERT_H

#include "Core/include/private/DImage.hxx"
#include "Core/include/DColor.h"

#include "NSTypes/RGB/include/DRGB.h"

namespace smil
{
  /**
   * @addtogroup AddonColor
   *
   * @{
   * @details
   * Color space conversion format conversion and @b gradient :
   * - RGB - <a href=https://en.wikipedia.org/wiki/RGB_color_space>RGB color
   * space</a>
   * - CIE LAB - <a href=https://en.wikipedia.org/wiki/CIELAB_color_space>CIE
   * LAB color space</a>
   * - CIE XYZ - <a href=https://fr.wikipedia.org/wiki/CIE_XYZ>CIE XYZ color
   * space</a>
   * - HSV - <a href=https://en.wikipedia.org/wiki/HSL_and_HSV>HSL and HSV color
   * space</a>
   * - Luminance
   *
   */

  /**
   * @brief RGBToXYZ
   * @param[in] imRgbIn : input Image
   * @param[out] imXyzOut : output Image
   */
  RES_T RGBToXYZ(const Image<RGB> &imRgbIn, Image<RGB> &imXyzOut);

  /**
   * @brief XYZToRGB
   * @param[in] imXyzIn : input Image
   * @param[out] imRgbOut : output Image
   */
  RES_T XYZToRGB(const Image<RGB> &imXyzIn, Image<RGB> &imRgbOut);

  /**
   * @brief XYZToLAB
   * @param[in] imXyzIn : input Image
   * @param[out] imLabOut : output Image
   */
  RES_T XYZToLAB(const Image<RGB> &imXyzIn, Image<RGB> &imLabOut);

  /**
   * @brief LABToXYZ
   * @param[in] imLabIn : input Image
   * @param[out] imXyzOut : output Image
   */
  RES_T LABToXYZ(const Image<RGB> &imLabIn, Image<RGB> &imXyzOut);

  /**
   * @brief RGBToHLS
   * @param[in] imRgbIn : input Image
   * @param[out] imHlsOut : output Image
   */
  RES_T RGBToHLS(const Image<RGB> &imRgbIn, Image<RGB> &imHlsOut);

  /**
   * @brief HLSToRGB
   * @param[in] imHlsIn : input Image
   * @param[out] imRgbOut : output Image
   */
  RES_T HLSToRGB(const Image<RGB> &imHlsIn, Image<RGB> &imRgbOut);

  /**
   * @brief RGBToHSV
   * @param[in] imRgbIn : input Image
   * @param[out] imHlsOut : output Image
   */
  RES_T RGBToHSV(const Image<RGB> &imRgbIn, Image<RGB> &imHlsOut);

  /**
   * @brief RGBToLAB
   * @param[in] imRgbIn : input Image
   * @param[out] imLabOut : output Image
   */
  RES_T RGBToLAB(const Image<RGB> &imRgbIn, Image<RGB> &imLabOut);

  /**
   * @brief LABToRGB
   * @param[in] imLabIn : input Image
   * @param[out] imRgbOut : output Image
   */
  RES_T LABToRGB(const Image<RGB> &imLabIn, Image<RGB> &imRgbOut);

  /**
   * @brief RGBToLuminance
   * @param[in] imRgbIn : input Image
   * @param[out] imLumOut : output Image
   */
  template <class T>
  RES_T RGBToLuminance(const Image<RGB> &imRgbIn, Image<T> &imLumOut)
  {
    ASSERT_ALLOCATED(&imRgbIn)
    ASSERT_SAME_SIZE(&imRgbIn, &imLumOut)

    ImageFreezer freeze(imLumOut);

    ImDtTypes<UINT8>::lineType      rArr      = imRgbIn.getPixels().arrays[0];
    ImDtTypes<UINT8>::lineType      gArr      = imRgbIn.getPixels().arrays[1];
    ImDtTypes<UINT8>::lineType      bArr      = imRgbIn.getPixels().arrays[2];
    typename ImDtTypes<T>::lineType outPixels = imLumOut.getPixels();

    size_t pixNbr = imRgbIn.getPixelCount();

    for (size_t i = 0; i < pixNbr; i++) {
      outPixels[i] = T(0.2126 * rArr[i] + 0.7152 * gArr[i] + 0.0722 * bArr[i]);
    }

    return RES_OK;
  }
  /** @} */
} // namespace smil

#endif // _D_COLOR_CONVERT_H
