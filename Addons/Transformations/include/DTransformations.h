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

#include "private/DResize.hpp"

#endif // __IM_TRANSFORMATIONS_H__
