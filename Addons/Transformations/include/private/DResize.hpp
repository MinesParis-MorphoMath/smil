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
 *   - 02/09/2020 - by Jose-Marcio Martins da Cruz
 *
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_IMAGE_RESIZE_HPP_
#define _D_IMAGE_RESIZE_HPP_

#include "Core/include/DCore.h"
#include "Base/include/DBase.h"
#include "Core/include/DErrors.h"

#include <string>

#ifdef USE_OPEN_MP
#include <omp.h>
#endif // USE_OPEN_MP

namespace smil
{
  /**
   * @addtogroup AddonTransformations Image Transformations
   * @{
   */

  /** @cond
   *
   */
  template <class T> class ImageResizeFuncNew
  {
  public:
    ImageResizeFuncNew(string method)
    {
      if (method != "linear" && method != "trilinear" && method != "closest")
        this->method = "trilinear";
      else
        this->method = method;
    }

    ImageResizeFuncNew()
    {
      method = "trilinear";
    }

    ~ImageResizeFuncNew()
    {
    }

    /*
     * Public resize members
     */
    RES_T resize(Image<T> &imIn, size_t width, size_t height, size_t depth,
                 Image<T> &imOut, string algorithm = "trilinear")
    {
      ASSERT_ALLOCATED(&imIn, &imOut)
      if (&imIn == &imOut) {
        Image<T> tmpIm(imIn, true);
        return resize(tmpIm, width, height, depth, imIn);
      }

      size_t dz = imIn.getDepth();
      if (algorithm == "auto") {
        if (isBinary(imIn))
          algorithm = "closest";
        else
          algorithm = dz > 1 ? "trilinear" : "bilinear";
      }

      if (algorithm == "closest") {
        if (dz > 1)
          return resize3DClosest(imIn, width, height, depth, imOut);
        else
          return resize2DClosest(imIn, width, height, imOut);
      }

      if (algorithm == "bilinear" || algorithm == "trilinear") {
        if (dz > 1)
          return resizeTrilinear(imIn, width, height, depth, imOut);
        else
          return resizeBilinear(imIn, width, height, imOut);
      }
      ERR_MSG("* Unknown algorithm " + algorithm);
      return RES_ERR;
    }

    RES_T resize(Image<T> &imIn, size_t width, size_t height, Image<T> &imOut,
                 string algorithm = "trilinear")
    {
      ASSERT_ALLOCATED(&imIn, &imOut)

      size_t depth = imIn.getDepth();
      return resize(imIn, width, height, depth, imOut, algorithm);
    }

    /*
     * Public scale member
     */
    RES_T scale(Image<T> &imIn, double kx, double ky, double kz,
                Image<T> &imOut, string algorithm = "trilinear")
    {
      ASSERT_ALLOCATED(&imIn, &imOut)

      if (&imIn == &imOut) {
        Image<T> tmpIm(imIn, true);
        return scale(tmpIm, kx, ky, kz, imIn);
      }

      size_t width  = imIn.getWidth();
      size_t height = imIn.getHeight();
      size_t depth  = imIn.getDepth();

      if (width > 1)
        width = max(1L, lround(kx * width));
      if (height > 1)
        height = max(1L, lround(ky * height));
      if (depth > 1)
        depth = max(1L, lround(kz * depth));

      return resize(imIn, width, height, depth, imOut, algorithm);
    }

    RES_T scale(Image<T> &imIn, size_t kx, size_t ky, Image<T> &imOut,
                string algorithm = "trilinear")
    {
      return scale(imIn, kx, ky, 1., imOut, algorithm);
    }

  private:
    /*
     *  ####   #        ####    ####   ######   ####    #####
     * #    #  #       #    #  #       #       #          #
     * #       #       #    #   ####   #####    ####      #
     * #       #       #    #       #  #            #     #
     * #    #  #       #    #  #    #  #       #    #     #
     *  ####   ######   ####    ####   ######   ####      #
     */
    /*
     * 2D - closest interpolation algorithm - naive loop implementation
     */
    RES_T resize2DClosest(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut)
    {
      size_t width  = imIn.getWidth();
      size_t height = imIn.getHeight();
      size_t depth  = imIn.getDepth();

      if (depth > 1) {
        return resize3DClosest(imIn, sx, sy, depth, imOut);
      }

      imOut.setSize(sx, sy, 1);
      ImageFreezer freeze(imOut);

      double cx = ((double) (width)) / sx;
      double cy = ((double) (height)) / sy;

      T *pixIn  = imIn.getPixels();
      T *pixOut = imOut.getPixels();

      for (size_t j = 0; j < sy; j++) {
#ifdef USE_OPEN_MP
        int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif // USE_OPEN_MP
        for (size_t i = 0; i < sx; i++) {
          size_t xo = round(cx * i);
          size_t yo = round(cy * j);

          xo = min(xo, width - 1);
          yo = min(yo, height - 1);

          T v                = pixIn[yo * width + xo];
          pixOut[j * sx + i] = v;
        }
      }

      return RES_OK;
    }

    /*
     * 3D - closest interpolation algorithm - naive loop implementation
     */
    RES_T resize3DClosest(Image<T> &imIn, size_t sx, size_t sy, size_t sz,
                          Image<T> &imOut)
    {
      size_t width  = imIn.getWidth();
      size_t height = imIn.getHeight();
      size_t depth  = imIn.getDepth();

      if (depth > 1) {
        return resize2DClosest(imIn, sx, sy, imOut);
      }
      imOut.setSize(sx, sy, sz);

      ImageFreezer freeze(imOut);

      double cx = ((double) (width)) / sx;
      double cy = ((double) (height)) / sy;
      double cz = ((double) (depth)) / sz;

      T *pixIn  = imIn.getPixels();
      T *pixOut = imOut.getPixels();

      for (size_t k = 0; k < sz; k++) {
        for (size_t j = 0; j < sy; j++) {
#ifdef USE_OPEN_MP
          int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif // USE_OPEN_MP
          for (size_t i = 0; i < sx; i++) {
            size_t xo = round(cx * i);
            size_t yo = round(cy * j);
            size_t zo = round(cz * k);

            xo = min(xo, width - 1);
            yo = min(yo, height - 1);
            zo = min(zo, depth - 1);

            T v = pixIn[(zo * depth + yo) * width + xo];
            pixOut[(k * sy + j) * sx + i] = v;
          }
        }
      }
      return RES_OK;
    }

    /*
     * #          #    #    #  ######    ##       #    #####   ######
     * #          #    ##   #  #        #  #      #    #    #  #
     * #          #    # #  #  #####   #    #     #    #    #  #####
     * #          #    #  # #  #       ######     #    #####   #
     * #          #    #   ##  #       #    #     #    #   #   #
     * ######     #    #    #  ######  #    #     #    #    #  ######
     */
    /*
     * 2D - Bilinear interpolation algorithm - naive loop implementation
     */
    RES_T resizeBilinear(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut)
    {
      off_t width  = imIn.getWidth();
      off_t height = imIn.getHeight();
      off_t depth  = imIn.getDepth();

      if (depth > 1) {
        return resizeTrilinear(imIn, sx, sy, depth, imOut);
      }

      imOut.setSize(sx, sy, 1);
      ImageFreezer freeze(imOut);

      double cx = ((double) (width)) / sx;
      double cy = ((double) (height)) / sy;

      T *pixIn  = imIn.getPixels();
      T *pixOut = imOut.getPixels();

      for (size_t j = 0; j < sy; j++) {
        off_t dy = width;
#ifdef USE_OPEN_MP
        int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif
        for (size_t i = 0; i < sx; i++) {
          Point<double> P(cx * i, cy * j, 0);

          off_t xl = (off_t) floor(P.x);
          off_t xh = (xl + 1) < width ? xl + 1 : xl;
          off_t yl = (off_t) floor(P.y);
          off_t yh = (yl + 1) < height ? yl + 1 : yl;

          double cxl = xh > xl ? (P.x - xl) : 1.;
          double cxh = 1. - cxl;
          double cyl = yh > yl ? (P.y - yl) : 1.;
          double cyh = 1. - cyl;

          T vp = pixIn[yl * dy + xl] * cxh * cyh +
                 pixIn[yl * dy + xh] * cxl * cyh +
                 pixIn[yh * dy + xl] * cxh * cyl +
                 pixIn[yh * dy + xh] * cxl * cyl;

          pixOut[j * sx + i] = vp;
        }
      }
      return RES_OK;
    }

    /*
     * 3D - Trilinear interpolation algorithm - naive loop implementation
     */
    RES_T resizeTrilinear(Image<T> &imIn, size_t sx, size_t sy, size_t sz,
                          Image<T> &imOut)
    {
      off_t width  = imIn.getWidth();
      off_t height = imIn.getHeight();
      off_t depth  = imIn.getDepth();

      if (depth == 1) {
        return resizeBilinear(imIn, sx, sy, imOut);
      }

      imOut.setSize(sx, sy, sz);
      ImageFreezer freeze(imOut);

      double cx = ((double) (width)) / sx;
      double cy = ((double) (height)) / sy;
      double cz = ((double) (depth)) / sz;

      T *pixIn  = imIn.getPixels();
      T *pixOut = imOut.getPixels();

      for (size_t k = 0; k < sz; k++) {
        off_t dz = width * height;
        for (size_t j = 0; j < sy; j++) {
          off_t dy = width;
#ifdef USE_OPEN_MP
          int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif
          for (size_t i = 0; i < sx; i++) {
            Point<double> P(cx * i, cy * j, cz * k);

            off_t xl = floor(P.x);
            off_t xh = (xl + 1) < width ? xl + 1 : xl;
            off_t yl = floor(P.y);
            off_t yh = (yl + 1) < height ? yl + 1 : yl;
            off_t zl = floor(P.z);
            off_t zh = (zl + 1) < depth ? zl + 1 : zl;

            double cxl = xh > xl ? (P.x - xl) : 1.;
            double cxh = 1. - cxl;
            double cyl = yh > yl ? (P.y - yl) : 1.;
            double cyh = 1. - cyl;
            double czl = zh > zl ? (P.z - zl) : 1.;
            double czh = 1. - czl;

            T vp = pixIn[zl * dz + yl * dy + xl] * cxh * cyh * czh +
                   pixIn[zl * dz + yl * dy + xh] * cxl * cyh * czh +
                   pixIn[zl * dz + yh * dy + xl] * cxh * cyl * czh +
                   pixIn[zl * dz + yh * dy + xh] * cxl * cyl * czh +
                   pixIn[zh * dz + yl * dy + xl] * cxh * cyh * czl +
                   pixIn[zh * dz + yl * dy + xh] * cxl * cyh * czl +
                   pixIn[zh * dz + yh * dy + xl] * cxh * cyl * czl +
                   pixIn[zh * dz + yh * dy + xh] * cxl * cyl * czl;

            pixOut[(k * sy + j) * sx + i] = vp;
          }
        }
      }
      return RES_OK;
    }

    string method;
  };
  /** @endcond */

  /** imageResize() - 3D image resize
   *
   * @details Resize a 3D image - the value of each pixel in the output image is
   * calculated from the input image after an interpolation algorithm.
   *
   * There are two available algorithms :
   * - @b closest - this is the simpler algorithm. Pixel values in the output
   * image are taken from the nearest corresponding pixel in the input image.
   * This algorithm doesn't increases the number of possible values. So, it must
   * be used when resizing @b binary images or images whose possible values
   * shall be preserved in the output image.
   * - @b trilinear (extension of @b bilinear algorithm for @b 3D images) - this
   * is the algorithm to use on @txtbold{gray level} images.
   *
   * @note
   * When algorithm is set to @b auto, the applied algorithm will be @b closest
   * for binary images and @b trilinear or @b bilinear for gray level images
   *
   * @param[in] imIn : input image
   * @param[in] sx, sy, sz : dimensions to be set on output image
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear, @b closest or @b auto.
   */
  template <typename T>
  RES_T imageResize(Image<T> &imIn, size_t sx, size_t sy, size_t sz,
                    Image<T> &imOut, string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFuncNew<T> func(algorithm);
    return func.resize(imIn, sx, sy, sz, imOut, algorithm);
  }

  /** imageResize() - 2D image resize
   *
   * @details Resize a 2D image - the value of each pixel in the output image is
   * calculated from the input image after an interpolation algorithm.
   *
   * @param[in] imIn : input image
   * @param[in] sx, sy : dimensions to be set on output image
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear ou @b closest.
   *
   * @overload
   */
  template <typename T>
  RES_T imageResize(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut,
                    string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    size_t depth = imOut.getDepth();

    ImageResizeFuncNew<T> func(algorithm);
    return func.resize(imIn, sx, sy, depth, imOut, algorithm);
  }

  /** imageResize() - 3D image resize
   *
   * @details Resize a 3D image - the value of each pixel in the output image is
   * calculated from the input image after an interpolation algorithm.
   *
   * The size of the output image is already set to what it should be.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear ou @b closest.
   *
   * @overload
   */
  template <typename T>
  RES_T imageResize(Image<T> &imIn, Image<T> &imOut,
                    string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    size_t width  = imOut.getWidth();
    size_t height = imOut.getHeight();
    size_t depth  = imOut.getDepth();

    ImageResizeFuncNew<T> func(algorithm);
    return func.resize(imIn, width, height, depth, imOut, algorithm);
  }

  /** imageScale() - 3D image scale (resize by a factor)
   *
   * @details 3D image scale - Scaling images is almost the same than resizing.
   Input parameters are the factors to multiply each dimension of the input
   image instead of the dimensions of output image.
   *
   * There are two available algorithms :
   * - @b closest - this is the simpler algorithm. Pixel values in the output
   * image are taken from the nearest corresponding pixel in the input image.
   * This algorithm doesn't increases the number of possible values. So, it must
   * be used when resizing @b binary images or images whose possible values
   * shall be preserved in the output image.
   * - @b trilinear (extension of @b bilinear algorithm for @b 3D images) - this
   * is the algorithm to use on @txtbold{gray level} images.
   *
   * @note
   * When algorithm is set to @b auto, the applied algorithm will be @b closest
   * for binary images and @b trilinear or @b bilinear for gray level images
   *
   * @param[in] imIn : input image
   * @param[in] kx, ky, kz : scale factors
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear, @b closest or @b auto.
   */
  template <typename T>
  RES_T imageScale(Image<T> &imIn, double kx, double ky, double kz,
                   Image<T> &imOut, string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFuncNew<T> func(algorithm);
    return func.scale(imIn, kx, ky, kz, imOut, algorithm);
  }

  /** imageScale() - 2D image scale
   *
   * @details 3D image scale - Scaling images is almost the same than resizing.
   * Input parameters are the factors to multiply each dimension of the input
   * image instead of the dimensions of output image.
   *
   * @param[in] imIn : input image
   * @param[in] kx, ky : scale factors
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear ou @b closest.
   */
  template <typename T>
  RES_T imageScale(Image<T> &imIn, double kx, double ky, Image<T> &imOut,
                   string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFuncNew<T> func(algorithm);
    return func.scale(imIn, kx, ky, 1., imOut, algorithm);
  }

  /** imageScale() - image scale (resize by a factor)
   *
   * @details 3D image scale - Scaling images is almost the same than resizing.   
   * Input parameters are the factors to multiply each dimension of the input
   * image instead of the dimensions of output image.
   *
   *
   * @param[in] imIn : input image
   * @param[in] k : scale factor applied to each axis.
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   trilinear (default), @b bilinear ou @b closest.
   */
  template <typename T>
  RES_T imageScale(Image<T> &imIn, double k, Image<T> &imOut,
                   string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFuncNew<T> func(algorithm);
    return func.scale(imIn, k, k, k, imOut, algorithm);
  }

  /******************************* */
  /*
   *  ####   #       #####
   * #    #  #       #    #
   * #    #  #       #    #
   * #    #  #       #    #
   * #    #  #       #    #
   *  ####   ######  #####
   */
  /** @cond */
  /** imageResizeClosest() - 3D image resize with a @txtbold{closest algorithm}
   *
   * @details Resize a 3D image - the value of each pixel in the output image is
   * taken from the corresponding closest pixel in the input image.
   *
   * @note
   * Use this function when resizing @b binary images or images with
   * a fixed number of levels. No interpolation is done and the number of
   * intensity levels doesn't increase.
   *
   * @param[in] imIn : input image
   * @param[in] sx, sy, sz : dimensions to be set on output image
   * @param[out] imOut : output image
   */
  template <typename T>
  RES_T imageResizeClosest(Image<T> &imIn, size_t sx, size_t sy, size_t sz,
                           Image<T> &imOut)
  {
#if 1
    return imageResize(imIn, sx, sy, sz, imOut, "closest");
#else
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFuncNew<T> func("closest");
    return func.resize(imIn, sx, sy, sz, imOut);
#endif
  }

  /** imageResizeClosest() - 2D image resize with a @txtbold{closest algorithm}
   *
   * @details Resize a 2D image - the value of each pixel in the output image is
   * taken from the corresponding closest pixel in the input image. pixel.
   *
   * @param[in] imIn : input image
   * @param[in] sx, sy : dimensions to be set on output image
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <typename T>
  RES_T imageResizeClosest(Image<T> &imIn, size_t sx, size_t sy,
                           Image<T> &imOut)
  {
#if 1
    return imageResize(imIn, sx, sy, imOut, "closest");
#else
    ASSERT_ALLOCATED(&imIn, &imOut)
    size_t depth = imOut.getDepth();

    ImageResizeFuncNew<T> func("closest");
    return func.resize(imIn, sx, sy, depth, imOut);
#endif
  }

  /** imageResizeClosest() - image resize with a @txtbold{closest algorithm}
   *
   * @details Resize a 3D image - the value of each pixel in the output image is
   * taken from the corresponding closest pixel in the input image.
   *
   * The size of the output image is already set to what it should be.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <typename T>
  RES_T imageResizeClosest(Image<T> &imIn, Image<T> &imOut)
  {
#if 1
    return imageResize(imIn, imOut, "closest");
#else
    ASSERT_ALLOCATED(&imIn, &imOut)
    size_t width  = imOut.getWidth();
    size_t height = imOut.getHeight();
    size_t depth  = imOut.getDepth();

    ImageResizeFuncNew<T> func("closest");
    return func.resize(imIn, width, height, depth, imOut);
#endif
  }

  /** imageScaleClosest() - 3D image scale (resize by a factor) with a
   * @txtbold{closest algorithm}
   * @details 3D image scale - the value of each pixel in the output image is
   * taken from the corresponding closest pixel in the input image.
   *
   * @note
   * Use this function when resizing @b binary images or images with
   * a fixed number of levels. No interpolation is done and the number of
   * intensity levels doesn't increase.
   *
   * @param[in] imIn : input image
   * @param[in] kx, ky, kz : scale factors
   * @param[out] imOut : output image
   */
  template <typename T>
  RES_T imageScaleClosest(Image<T> &imIn, double kx, double ky, double kz,
                          Image<T> &imOut)
  {
#if 1
    return imageScale(imIn, kx, ky, kz, imOut, "closest");
#else
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFuncNew<T> func("closest");
    return func.scale(imIn, kx, ky, kz, imOut);
#endif
  }

  /** imageScaleClosest() - 2D image scale (resize by a factor) with a
   * @txtbold{closest algorithm}
   * @details 2D image scale - the value of each pixel in the output image is
   * taken from the corresponding closest pixel in the input image.
   *
   * @param[in] imIn : input image
   * @param[in] kx, ky : scale factors
   * @param[out] imOut : output image
   */
  template <typename T>
  RES_T imageScaleClosest(Image<T> &imIn, double kx, double ky, Image<T> &imOut)
  {
#if 1
    return imageScale(imIn, kx, ky, imOut, "closest");
#else
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFuncNew<T> func("closest");
    return func.scale(imIn, kx, ky, 1., imOut);
#endif
  }

  /** imageScaleClosest() - image scale (resize by a factor) with a
   * @txtbold{closest algorithm}
   * @details image scale - the value of each pixel in the output image is
   * taken from the corresponding closest pixel in the input image.
   *
   *
   * @param[in] imIn : input image
   * @param[in] k : scale factor applied to each axis.
   * @param[out] imOut : output image
   */
  template <typename T>
  RES_T imageScaleClosest(Image<T> &imIn, double k, Image<T> &imOut)
  {
#if 1
    return imageScale(imIn, k, k, k, imOut, "closest");
#else
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFuncNew<T> func("closest");
    return func.scale(imIn, k, k, k, imOut);
#endif
  }
  /** @endcond */
  /** @} */
} // namespace smil

#endif // _D_IMAGE_RESIZE_HPP_
