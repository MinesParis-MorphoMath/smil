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

#ifndef _D_IMAGE_ROTATE_HPP_
#define _D_IMAGE_ROTATE_HPP_

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

  template <class T> class ImageRotate
  {
  public:
    ImageRotate()
    {
    }

    ~ImageRotate()
    {
    }

    RES_T Rotate(Image<T> &imIn, int angle, Image<T> &imOut)
    {
      angle = angle % 360;

      if (angle % 90 == 0)
        return RotateX90(imIn, angle, imOut);

      if (angle % 90 != 0) {
        ERR_MSG("Rotation angle shall be a multiple of 90 degres");
        return RES_ERR;
      }

      return RES_OK;
    }

  private:
    RES_T RotateX90(Image<T> &imIn, int angle, Image<T> &imOut)
    {
      RES_T result = RES_OK;

      ASSERT_ALLOCATED(&imIn, &imOut);

      angle = angle % 360;

      if (angle % 90 != 0) {
        ERR_MSG("Rotation angle shall be a multiple of 90 degres");
        return RES_ERR;
      }

      if (angle == 0) {
        result = copy(imIn, imOut);
        imOut.modified();
        return result;
      }

      ImageFreezer freeze(imOut);

      off_t w = imIn.getWidth();
      off_t h = imIn.getHeight();
      off_t d = imIn.getDepth();

      if (angle == 90 || angle == 270) {
        imOut.setSize(h, w, d);
      } else {
        imOut.setSize(w, h, d);
      }

      typedef typename ImDtTypes<T>::lineType lineType;
      lineType pixIn  = imIn.getPixels();
      lineType pixOut = imOut.getPixels();

      switch (angle) {
      case 90:
#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
        for (off_t k = 0; k < d; k++) {
          off_t offset = k * w * h;
          T *sIn       = (T *) (pixIn + offset);
          T *sOut      = (T *) (pixOut + offset);
          for (off_t j = 0; j < h; j++) {
            for (off_t i = 0; i < w; i++) {
              sOut[i * h + (w - 1 - j)] = sIn[j * w + i];
            }
          }
        }
        break;
      case 180:
#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
        for (off_t k = 0; k < d; k++) {
          off_t offset = k * w * h;
          T *sIn       = (T *) (pixIn + offset);
          T *sOut      = (T *) (pixOut + offset);
          for (off_t j = 0; j < h; j++) {
            for (off_t i = 0; i < w; i++) {
              sOut[(h - 1 - j) * w + (w - 1 - i)] = sIn[j * w + i];
            }
          }
        }
        break;
      case 270:
#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
        for (off_t k = 0; k < d; k++) {
          off_t offset = k * w * h;
          T *sIn       = (T *) (pixIn + offset);
          T *sOut      = (T *) (pixOut + offset);
          for (off_t j = 0; j < h; j++) {
            for (off_t i = 0; i < w; i++) {
              sOut[(w - 1 - i) * h + j] = sIn[j * w + i];
            }
          }
        }
        break;
      default:
        break;
      }

      imOut.modified();
      return RES_OK;
    }
  };

  /*
   *
   */
  template <class T>
  RES_T imageRotate(Image<T> &imIn, int angle, Image<T> &imOut)
  {
    ImageRotate<T> imr;

    if (&imIn == &imOut) {
      Image<T> imTmp(imIn, true);
      return imageRotate(imTmp, angle, imOut);
    }

    return imr.Rotate(imIn, angle, imOut);
  }


  /*
   * First primitive version
   */
  template <class T>
  RES_T XimageRotate(Image<T> &imIn, int angle, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    RES_T result = RES_OK;

    angle = angle % 360;

    if (angle % 90 != 0) {
      ERR_MSG("Rotation angle shall be a multiple of 90 degres");
      return RES_ERR;
    }

    ImageFreezer freeze(imOut);

    switch (angle) {
    case 0:
      result = copy(imIn, imOut);
      break;
    case 90:
      result = imageTranspose(imIn, imOut);
      if (result == RES_OK)
        result = imageHorizFlip(imOut, imOut);
      break;
    case 180:
      result = imageVertFlip(imIn, imOut);
      if (result == RES_OK)
        result = imageHorizFlip(imOut, imOut);
      break;
    case 270:
      result = imageHorizFlip(imIn, imOut);
      if (result == RES_OK)
        result = imageTranspose(imOut, imOut);
      break;
    default:
      result = RES_ERR;
      break;
    }
    imOut.modified();
    return result;
  }
} // namespace smil

#endif // _D_IMAGE_ROTATE_HPP_
