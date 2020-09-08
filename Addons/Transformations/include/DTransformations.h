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
 *
 *
 * History :
 *   - 08/06/2020 - by Jose-Marcio Martins da Cruz
 *
 *
 * __HEAD__ - Stop here !
 */

#ifndef __IM_TRANSFORMATIONS_H__
#define __IM_TRANSFORMATIONS_H__

#include "Core/include/DCore.h"
#include "Base/include/DBase.h"

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup AddonTransformations Image Transformations
   * @{
   *
   *   @brief Common useful image transformations
   *
   *   @details
   *   Some image transformations functions. Most of these calls are intended
   *   to be included in the main library
   *
   *   @author Jose-Marcio Martins da Cruz (port to Smil)
   *
   *
   */

  /**
   * @brief mageTranspose() : 3D image transposition
   *
   * Transpose 3D images. The image is mirrored with respect to the main
   * diagonal and the shape is adapted.
   *
   * The order parameter defines where to redirect input axes.
   * E.g. : if @b order is @b xzy, axes @b y and @b z will be exchanged.
   *
   * Possible values for parameter @b order are : <b> xyz, xzy, yxz, yzx, zxy,
   * zyx, xy, yx</b> and an empty string. Notice that @b xyz and @b xy, does
   * nothing but just copies input image into output image.
   *
   * @param[in]  imIn : input Image
   * @param[in]  order : axis order in the output image
   * @param[out] imOut : output Image
   *
   */
  template <class T>
  RES_T imageTranspose(const Image<T> &imIn, Image<T> &imOut,
                       const string order = "yxz");

  /**
   * @brief imageTranspose() : 3D image transposition
   *
   * @param[in,out]  im : input/output Image
   * @param[in]  order : axis order in the output image
   *
   * @overload
   */
  template <class T>
  RES_T imageTranspose(Image<T> &im, const string order = "yxz")
  {
    return imageTranspose(im, im, order);
  }

  /**
   * @brief imageVertFlip() : Vertical Flip
   *
   * Mirror an image using an horizontal line (or plan for 3D images) in the
   * center of the image.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   */
  template <class T> RES_T imageVertFlip(Image<T> &imIn, Image<T> &imOut);

  /**
   * @brief imageVertFlip() : Vertical Flip
   *
   * @param[in,out]  im : input/output Image
   *
   * @overload
   */
  template <class T> RES_T imageVertFlip(Image<T> &im)
  {
    return imageVertFlip(im, im);
  }

  /**
   * @brief imageHorizFlip() : Horizontal Flip
   *
   * Mirror an image using a vertical line (or plan for 3D images) in the
   * center of the image.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   */
  template <class T> RES_T imageHorizFlip(Image<T> &imIn, Image<T> &imOut);

  /**
   * @brief imageHorizFlip() : Horizontal Flip
   *
   * @param[in,out]  im : input/output Image
   *
   * @overload
   */
  template <class T> RES_T imageHorizFlip(Image<T> &im)
  {
    return imageHorizFlip(im, im);
  }

  /**
   * @brief imageRotate() : Rotate an image by an angle multiple of 90 degres
   *
   * @param[in]  imIn : input Image
   * @param[in]  angle : rotation angle (0, 90, 180 or 270, 360, 450, ...) in
   * degres
   * @param[out] imOut : output Image
   *
   * @note
   * - If @b angle equals @b 0, just copy the input image into the output image.
   * @note
   * - When calling this function on @b 3D images, all slices are rotated the
   * same way than the first slice.
   */
  template <class T>
  RES_T imageRotate(Image<T> &imIn, int angle, Image<T> &imOut);

  /**
   * @brief imageRotate() : Rotate an image by an angle multiple of 90 degres
   *
   * @param[in,out]  im : input/output Image
   * @param[in]  angle : rotation angle (0, 90, 180 or 270, 360, 450, ...) in
   * degres
   *
   * @overload
   */
  template <class T> RES_T imageRotate(Image<T> &im, int angle)
  {
    return imageRotate(im, angle, im);
  }

  /** @cond */
  template <class T>
  RES_T imageScale(const Image<T> &imIn, float kWidth, float kHeight,
                   float KDepth, Image<T> &imOut);

  template <class T>
  RES_T imageScale(const Image<T> &imIn, float kWidth, float kHeight,
                   Image<T> &imOut);

  template <class T>
  RES_T imageScale(const Image<T> &imIn, float k, Image<T> &imOut);

  template <class T>
  RES_T imageResize(const Image<T> &imIn, size_t width, size_t height,
                    size_t depth, Image<T> &imOut);

  template <class T>
  RES_T imageResize(const Image<T> &imIn, size_t width, size_t height,
                    Image<T> &imOut);
  /** @endcond */

  /** @} */

} // namespace smil

#include "private/DFlip.hpp"
#include "private/DRotate.hpp"
#include "private/DResize.hpp"

#endif // __IM_TRANSFORMATIONS_H__
