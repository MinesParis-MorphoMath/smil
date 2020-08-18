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

#ifndef _D_TRANSPOSE_HPP_
#define _D_TRANSPOSE_HPP_

#include "Core/include/DCore.h"
#include "Base/include/DBase.h"
#include "Core/include/DErrors.h"

#include <string>

#ifdef USE_OPEN_MP
#include <omp.h>
#endif // USE_OPEN_MP

namespace smil
{
  /*
   *
   */

  inline bool getTransposeLut(string order, int lut[])
  {
    if (order == "yxz" || order == "yx") {
      lut[0] = 1;
      lut[1] = 0;
      lut[2] = 2;
      return true;
    }
    if (order == "xzy") {
      lut[0] = 0;
      lut[1] = 2;
      lut[2] = 1;
      return true;
    }
    if (order == "yzx") {
      lut[0] = 1;
      lut[1] = 2;
      lut[2] = 0;
      return true;
    }
    if (order == "zxy") {
      lut[0] = 2;
      lut[1] = 0;
      lut[2] = 1;
      return true;
    }
    if (order == "zyx") {
      lut[0] = 2;
      lut[1] = 1;
      lut[2] = 0;
      return true;
    }
    return false;
  }

  template <class T>
  RES_T imageTranspose(const Image<T> &imIn, Image<T> &imOut, string order)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);

    if (&imIn == &imOut) {
      Image<T> imTmp(imIn, true);
      return imageTranspose(imTmp, imOut, order);
    }

    ImageFreezer freeze(imOut);

    size_t szIn[3], szOut[3];
    int lut[3];

    if (order == "xyz" || order == "xy") {
      imOut = Image<T>(imIn, true);
      return RES_OK;
    }

    if (order == "")
      order = "yxz";

    if (!getTransposeLut(order, lut)) {
      ERR_MSG("Wrong value for parameter order");
      return RES_ERR;
    }
#if 0
    if (order == "yxz" || order == "yx") {
      lut[0] = 1;
      lut[1] = 0;
      lut[2] = 2;
    }
    if (order == "xzy") {
      lut[0] = 0;
      lut[1] = 2;
      lut[2] = 1;
    }
    if (order == "yzx") {
      lut[0] = 1;
      lut[1] = 2;
      lut[2] = 0;
    }
    if (order == "zxy") {
      lut[0] = 2;
      lut[1] = 0;
      lut[2] = 1;
    }
    if (order == "zyx") {
      lut[0] = 2;
      lut[1] = 1;
      lut[2] = 0;
    }
#endif

    imIn.getSize(szIn);
    for (int i = 0; i < 3; i++)
      szOut[i] = szIn[lut[i]];
    int r = imOut.setSize(szOut);
    if (r != RES_OK) {
      ERR_MSG("Can't set imOut size");
    }
    {
      size_t ix[3];
      T pixVal;

#ifdef USE_OPEN_MP
      // int nthreads = Core::getInstance()->getNumberOfThreads();
      // int nthreads = omp_get_num_threads(void);
#pragma omp parallel private(ix, pixVal)
#endif // USE_OPEN_MP

      {
        for (ix[2] = 0; ix[2] < szIn[2]; ix[2]++) {
          for (ix[1] = 0; ix[1] < szIn[1]; ix[1]++) {
            for (ix[0] = 0; ix[0] < szIn[0]; ix[0]++) {
              pixVal = imIn.getPixel(ix[0], ix[1], ix[2]);
              imOut.setPixel(ix[lut[0]], ix[lut[1]], ix[lut[2]], pixVal);
            }
          }
        }
      }
    }
    imOut.modified();
    return RES_OK;
  }

} // namespace smil

#endif // _D_TRANSPOSE_HPP_
