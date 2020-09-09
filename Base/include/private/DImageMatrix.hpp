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

#ifndef _D_IMAGE_MATRIX_HPP
#define _D_IMAGE_MATRIX_HPP

#include "Core/include/private/DImage.hpp"
#include "Core/include/DErrors.h"

namespace smil
{
  /**
   * @ingroup Base
   * @defgroup Matrix Matrix operations
   *
   * Matrix operations on images : multiplication and transposition
   * @{
   */

  /** @cond */
  template <class T> class ImageTransposeFunc
  {
  private:
    vector<int> lut{1, 0, 2};

    bool setOrder(string order)
    {
      lut.resize(3);
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

  public:
    ImageTransposeFunc()
    {
      setOrder("yxz");
    }
    ImageTransposeFunc(string order)
    {
      if (!setOrder(order))
        ERR_MSG("Unknown transpose order " + order);
    }

    RES_T transpose(const Image<T> &imIn, Image<T> &imOut, string order)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);

      if (&imIn == &imOut) {
        Image<T> imTmp(imIn, true);
        return transpose(imTmp, imOut, order);
      }

      size_t szIn[3], szOut[3];

      if (order == "xyz" || order == "xy") {
        imOut = Image<T>(imIn, true);
        return RES_OK;
      }

      if (order == "")
        order = "yxz";

      if (!setOrder(order)) {
        ERR_MSG("Wrong value for parameter order");
        return RES_ERR;
      }

      imIn.getSize(szIn);
      for (int i = 0; i < 3; i++)
        szOut[i] = szIn[lut[i]];

      ASSERT(imOut.setSize(szOut) == RES_OK);

      ImageFreezer freeze(imOut);
      {
        size_t ix[3];

#ifdef USE_OPEN_MP
        int nthreads = Core::getInstance()->getNumberOfThreads();
#  pragma omp parallel private(ix) num_threads(nthreads)
#endif // USE_OPEN_MP
        for (ix[2] = 0; ix[2] < szIn[2]; ix[2]++) {
          for (ix[1] = 0; ix[1] < szIn[1]; ix[1]++) {
            for (ix[0] = 0; ix[0] < szIn[0]; ix[0]++) {
              T pixVal = imIn.getPixel(ix[0], ix[1], ix[2]);
              imOut.setPixel(ix[lut[0]], ix[lut[1]], ix[lut[2]], pixVal);
            }
          }
        }
      }
      imOut.modified();
      return RES_OK;
    }
  };
  /** @endcond */
  
  /**
   * @brief matTranspose() : 3D image transposition
   *
   * Transpose 3D images. The image is mirrored with respect to the main
   * diagonal and the shape is adapted.
   *
   * The order parameter defines where to redirect input axes.
   * E.g. : if @b order is @b xzy, axes @b y and @b z will be exchanged.
   *
   * Possible values for parameter @b order are : <b> xyz, xzy, yxz, yzx, zxy,
   * zyx, xy, yx</b> and an empty string.
   * @note
   * - @b xyz and @b xy, does nothing but just copies input image into output 
   *   image.
   * - @b yxz or @b yx correspond to the usual transposition of @b 2D matrices.
   *   When applied to @b 3D images, all slices are transposed. 
   *
   * @param[in]  imIn : input Image
   * @param[in]  order : axis order in the output image
   * @param[out] imOut : output Image
   *
   */
  template <class T>
  RES_T matTranspose(const Image<T> &imIn, Image<T> &imOut, string order = "yxz")
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ImageTransposeFunc<T> tmat(order);
    return tmat.transpose(imIn, imOut, order);
  }

  /**
   * @brief matTranspose() : 3D image transposition
   *
   * @param[in,out]  im : input/output Image
   * @param[in]  order : axis order in the output image
   *
   * @overload
   */
  template <class T>
  RES_T matTranspose(Image<T> &im, const string order = "yxz")
  {
    ASSERT_ALLOCATED(&im);
    ImageTransposeFunc<T> tmat(order);
    return tmat.transpose(im, im, order);
  }

  /**
   * matMultiply() - Matrix multiplication (for now, only in 2D)
   *
   * @param[in] imIn1 : input image
   * @param[in] imIn2 : input image
   * @param[out] imOut : output image
   *
   * @vectorized
   * @parallelized
   */
  template <class T>
  RES_T matMultiply(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn1, &imIn2);
    size_t size1[3], size2[3];
    imIn1.getSize(size1);
    imIn2.getSize(size2);

    if (size1[2] != 1 || size2[2] != 1)
      return RES_ERR_NOT_IMPLEMENTED;

    ImageFreezer freezer(imOut);

    // Verify that the number of columns m in imIn1 is equal to the number of
    // rows m in imIn2
    ASSERT((size1[0] == size2[1]), "Wrong matrix sizes!", RES_ERR);
    ASSERT((imOut.setSize(size2[0], size1[1]) == RES_OK));

    Image<T> transIm(size2[1], size2[0]);

    // Transpose imIn2 matrix to allow vectorization
    ASSERT((matTranspose(imIn2, transIm) == RES_OK));

    typedef typename ImDtTypes<T>::sliceType sliceType;
    typedef typename ImDtTypes<T>::lineType lineType;

    sliceType lines    = imIn1.getLines();
    sliceType outLines = imOut.getLines();
    sliceType cols     = transIm.getLines();
    lineType line;
    lineType outLine;
    lineType col;

    size_t y;

#ifdef USE_OPEN_MP
    int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel private(line, outLine, col)
#endif // USE_OPEN_MP
    {
#ifdef USE_OPEN_MP
#pragma omp for schedule(dynamic, nthreads) nowait
#endif // USE_OPEN_MP
      for (y = 0; y < size1[1]; y++) {
        line    = lines[y];
        outLine = outLines[y];
        for (size_t x = 0; x < size2[0]; x++) {
          col      = cols[x];
          T outVal = 0;
          for (size_t i = 0; i < size1[0]; i++)
            outVal += line[i] * col[i];
          outLine[x] = outVal;
        }
      }
    }

    return RES_OK;
  }

  /** @}*/

} // namespace smil

#endif // _D_IMAGE_MATRIX_HPP
