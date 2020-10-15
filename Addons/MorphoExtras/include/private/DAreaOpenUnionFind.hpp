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

#ifndef _D_AREA_OPEN_UNION_FIND_HPP
#define _D_AREA_OPEN_UNION_FIND_HPP

#include <cmath>
#include <cstring>
#include "Core/include/DCore.h"

using namespace std;

namespace smil
{
  /**
   * @addtogroup XXX
   *   or
   * @defgroup   XXX YYY
   *   or
   * @ingroup    XXX
   *
   * @{ */

  template <typename T1, typename T2>
  class AreaOpenUnionFind
  {
  public:
    RES_T AreaOpen(const Image<T1> imIn, size_t size, Image<T2> imOut)
    {
      return RES_OK;
    }

  private:
    size_t width;
    size_t height;
    size_t depth;

    map<T1, vector<off_t>> qPix;

    void buildMap(const Image<T1> imIn)
    {
      typename Image<T1>::lineType pixels = imIn.getPixels();
      for (off_t i = 0; i < imIn.getPixelCount(); i++) {
        T1 val = pixels[i];
        qPix[val].push_back(i);

        size_t capacity = qPix[val].capacity();
        if ((capacity - qPix[val].size()) < 1024)
          qPix[val].reserve(capacity + 1024);
      }
    }
  };

  /** @} */

} // namespace smil

#endif // _D_AREA_OPEN_UNION_FIND_HPP
