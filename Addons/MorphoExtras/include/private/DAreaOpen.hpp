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
 *   Area Open using UnionFind Method
 *
 * History :
 *   - 15/10/2020 - by Jose-Marcio Martins da Cruz
 *     Porting from xxx
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_AREA_OPEN_HPP
#define _D_AREA_OPEN_HPP

#include <cmath>
#include <cstring>
#include "Core/include/DCore.h"

#include "DAreaOpenUnionFind.hpp"

namespace smil
{
  /**
   * @ingroup  AddonMorphoExtras
   * @defgroup AddonMorphoExtrasAttrOpen      Attribute Open/Close
   *
   * @{
   */

  /**
   * areaOpening() -
   *
   * @param[in] imIn : input image
   * @param[in] size : area threshold to stop
   * @param[out] imOut : output image
   * @param[in] se: structuring element
   * @param[in] method : algorithm. Nowadays, only "unionfind" is available
   */
  template <typename T>
  RES_T areaOpening(const Image<T> &imIn, size_t size, Image<T> &imOut,
                    StrElt &se = DEFAULT_SE, const string method = "unionfind")
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    if (method == "unionfind") {
      UnionFindFunctions<T> uff;
      return uff.areaOpen(imIn, size, imOut, se);
    }

    cout << "This method isn't implemented : " << method << endl;
    return RES_ERR;
  }

  /**
   * areaClosing() -
   *
   * @param[in] imIn : input image
   * @param[in] size : area threshold to stop
   * @param[out] imOut : output image
   * @param[in] se: structuring element
   * @param[in] method : algorithm. Nowadays, only "unionfind" is available
   */
  template <typename T>
  RES_T areaClosing(const Image<T> &imIn, size_t size, Image<T> &imOut,
                    StrElt &se = DEFAULT_SE, const string method = "unionfind")
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    if (method == "unionfind") {
      Image<T> imTmp(imIn);
      RES_T res = inv(imIn, imTmp);
      if (res == RES_OK) {
        UnionFindFunctions<T> uff;
        res = uff.areaOpen(imTmp, size, imOut, se);
      }
      if (res == RES_OK)
        res = inv(imOut, imOut);
      return res;
    }

    cout << "This method isn't implemented : " << method << endl;
    return RES_ERR;
  }

  /**
   * @}
   */
} // namespace smil

#endif // _D_AREA_OPEN_UNION_FIND_HPP
