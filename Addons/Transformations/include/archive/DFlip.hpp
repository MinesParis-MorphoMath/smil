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

#ifndef _D_IMAGE_FLIP_HPP_
#define _D_IMAGE_FLIP_HPP_

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
  template <class T> RES_T vertFlip(Image<T> &imIn, Image<T> &imOut)
  {
    if (&imIn == &imOut) {
      Image<T> imTmp = Image<T>(imIn, true);
      return horizFlip(imTmp, imOut);
    }

    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    typename Image<T>::sliceType *slicesIn  = imIn.getSlices();
    typename Image<T>::sliceType *slicesOut = imOut.getSlices();
    typename Image<T>::sliceType linesIn;
    typename Image<T>::sliceType linesOut;

    size_t width  = imIn.getWidth();
    size_t height = imIn.getHeight();
    size_t depth  = imIn.getDepth();

#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
    for (size_t k = 0; k < depth; k++) {
      linesIn  = slicesIn[k];
      linesOut = slicesOut[k];

      for (size_t j = 0; j < height; j++)
        copyLine<T>(linesIn[j], width, linesOut[height - 1 - j]);
    }

    imOut.modified();

    return RES_OK;
  }

  /*
   *
   */
  template <class T> RES_T horizFlip(Image<T> &imIn, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    if (&imIn == &imOut) {
      Image<T> imTmp = Image<T>(imIn, true);
      return horizFlip(imTmp, imOut);
    }

    typename Image<T>::sliceType linesIn  = imIn.getLines();
    typename Image<T>::sliceType linesOut = imOut.getLines();

    size_t width  = imIn.getWidth();
    size_t height = imIn.getHeight();
    size_t depth  = imIn.getDepth();

#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
    for (size_t k = 0; k < height * depth; k++) {
      T *lIn  = linesIn[k];
      T *lOut = linesOut[k];

      for (size_t j = 0; 2 * j < width; j++) {
        lOut[j]             = lIn[width - 1 - j];
        lOut[width - 1 - j] = lIn[j];
      }
    }

    imOut.modified();

    return RES_OK;
  }

} // namespace smil

#endif // _D_IMAGE_FLIP_HPP_
