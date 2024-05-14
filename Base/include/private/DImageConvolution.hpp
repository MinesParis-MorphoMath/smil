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

#ifndef _D_IMAGE_CONVOLUTION_HPP
#define _D_IMAGE_CONVOLUTION_HPP

#include "DLineArith.hpp"
#include "Core/include/private/DBufferPool.hpp"

#include <cmath>

namespace smil
{
  /**
   * @ingroup Base
   * @defgroup Convolution Convolution
   *
   * @b 2D and 3D Convolution with linear kernels
   *
   * @{
   */

  /** @cond */
  template <typename T>
  class GaussianFilterFunct
  {
  private:
    vector<double> kernel;
    int            radius;
    double         sigma;

    void setupKernel()
    {
      kernel.resize(2 * radius + 1);
      if (radius == 0)
        sigma = 1.;

      double sum = 0;
      for (int i = -radius; i <= radius; i++) {
        kernel[i + radius] = std::exp(-(std::pow(i / sigma, 2) / 2));
        sum += kernel[i + radius];
      }
      for (int i = -radius; i <= radius; i++) {
        kernel[i + radius] /= sum;
        //cout << "kernel " << i << " " << kernel[i + radius] << "\n";
      }
    }

    double getKernelValue(int i)
    {
      if ((i < -radius) || (i > radius))
        return 0.;

      return kernel[i + radius];
    }

  public:
    GaussianFilterFunct(int radius, double sigma) : radius(radius), sigma(sigma)
    {
      setupKernel();
    }

    GaussianFilterFunct(int radius = 2) : radius(radius)
    {
      if (radius > 0)
        sigma = radius / 2.;
      else
        sigma = 1.;
      setupKernel();
    }

    ~GaussianFilterFunct()
    {
    }

    RES_T Convolve(Image<T> &imIn, int radius, Image<T> &imOut)
    {
      this->radius = radius;
      this->sigma  = radius / 2.;
      setupKernel();

      return this->_do_Convolve(imIn, imOut);
    }
  private:
    RES_T _do_Convolve(Image<T> &imIn, Image<T> &imOut)
    {
      typename ImDtTypes<T>::lineType in  = imIn.getPixels();
      typename ImDtTypes<T>::lineType out = imOut.getPixels();

      off_t W = imIn.getWidth();
      off_t H = imIn.getHeight();
      off_t D = imIn.getDepth();

      //size_t nbPixels = imIn.getPixelCount();

      Image<T>                        imTmp(imIn);
      typename ImDtTypes<T>::lineType tmp = imTmp.getPixels();

#ifdef USE_OPEN_MP
      int nthreads = Core::getInstance()->getNumberOfThreads();
#endif // USE_OPEN_MP

      /*
       * convolution in X
       */
      //vector<T> outX(nbPixels, 0);
      double sk = 0.;
#ifdef USE_OPEN_MP
#pragma omp parallel num_threads(nthreads) private(sk)
#endif // USE_OPEN_MP
      {
        for (off_t z = 0; z < D; z++) {
          for (off_t y = 0; y < H; y++) {
#ifdef USE_OPEN_MP
#pragma omp for simd
#endif // USE_OPEN_MP
            for (off_t x = 0; x < W; x++) {
              off_t  i0   = (z * H + y) * W + x;
              double sumV = 0.;

              sk = 0.;
              for (off_t i = -radius; i <= radius; i++) {
                if ((x + i < 0) || (x + i > W - 1))
                  continue;
                double valK = getKernelValue(i);
                sk += valK;
                sumV += in[i0 + i] * valK;
              }
              //sk = 1.;
              out[i0] = T(round(sumV / sk));
            }
          }
        }
      }

      /*
       * convolution in Y
       */
      copy(imOut, imTmp);
#ifdef USE_OPEN_MP
#pragma omp parallel num_threads(nthreads) private(sk)
#endif // USE_OPEN_MP
      {
        off_t stride = W;

        for (off_t z = 0; z < D; z++) {
          for (off_t y = 0; y < H; y++) {
#ifdef USE_OPEN_MP
#pragma omp for simd
#endif // USE_OPEN_MP
            for (off_t x = 0; x < W; x++) {
              off_t  i0   = (z * H + y) * W + x;
              double sumV = 0.;

              sk = 0.;
              for (off_t i = -radius; i <= radius; i++) {
                if ((y + i < 0) || (y + i > H - 1))
                  continue;
                double valK = getKernelValue(i);
                sk += valK;
                sumV += tmp[i0 + i * stride] * valK;
              }
              //sk = 1.;
              out[i0] = T(round(sumV / sk));
            }
          }
        }
      }

      /*
       * convolution in Z
       */
      if (D > 1) {
        copy(imOut, imTmp);
#ifdef USE_OPEN_MP
#pragma omp parallel num_threads(nthreads)
#endif // USE_OPEN_MP
        {
          off_t stride = W * H;
          for (off_t z = 0; z < D; z++) {
            for (off_t y = 0; y < H; y++) {
#ifdef USE_OPEN_MP
#pragma omp for simd
#endif // USE_OPEN_MP
              for (off_t x = 0; x < W; x++) {
                off_t  i0   = (z * H + y) * W + x;
                double sumV = 0.;

                sk = 0.;
                for (off_t i = -radius; i <= radius; i++) {
                  if ((z + i < 0) || (z + i > D - 1))
                    continue;
                  double valK = getKernelValue(i);
                  sk += valK;
                  sumV += tmp[i0 + i * stride] * valK;
                }
                out[i0] = T(round(sumV / sk));
              }
            }
          }
        }
      }
      return RES_OK;
    }
  };
  /** @endcond */

  /**
   * @b gaussianFilter() - @b 3D Gaussian filter
   *
   * Gaussian Filter : convolution de l'input image against a Gaussian Kernel
   * of size <b><c>2 * radius + 1</c></b>
   *
   * The kernel is defined, in each direction, by :
   * @f[
   *  kernel[i] = exp(- \frac{i^2}{2 . \sigma^2}),
   *  \; i \in [-radius, +radius],
   *  \;\sigma = radius / 2
   * @f]
   *
   * @param[in] imIn : input image
   * @param[in] radius : radius of the gaussian kernel
   * @param[out] imOut : output image
   *
   * @smilexample{example-gaussian-filter.py}
   */
  template <class T>
  RES_T gaussianFilter(Image<T> &imIn, int radius, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    if (&imIn == &imOut) {
      Image<T> tmpIm(imIn, true);
      return gaussianFilter(tmpIm, radius, imOut);
    }

    GaussianFilterFunct<T> k;

    ImageFreezer freeze(imOut);
    return k.Convolve(imIn, radius, imOut);
  }

  /** @cond */
  /**
   * 2D Gaussian filter
   *
   * The size of the filter is 2*radius+1
   */
  template <class T>
  RES_T gaussianFilterOrg(const Image<T> &imIn, size_t radius, Image<T> &imOut)
  {
    CHECK_ALLOCATED(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    int            kernelSize = radius * 2 + 1;
    vector<double> kernel(kernelSize);

    double sigma = double(radius) / 2.;
    double sum   = 0.;

    //  Determine kernel coefficients
    for (int i = 0; i < kernelSize; i++) {
      kernel[i] = double(std::exp(-std::pow((i - radius) / sigma, 2) / 2.));
      sum += kernel[i];
    }
    // Normalize
    for (int i = 0; i < kernelSize; i++)
      kernel[i] /= sum;

    return convolve(imIn, kernel, imOut);
  }
  /** @endcond */

  /**
   * horizConvolve() - 2D Horizontal convolution
   *
   * 2D horizontal convolution using a @TB{1D kernel}
   *
   * @param[in] imIn : input image
   * @param[in] kernel : an <b>1D kernel</b> (as a vector)
   * @param[out] imOut : output image
   *
   * @b Example:
   * @code{.py}
   * import smilPython as sp
   *
   * im1 = sp.Image("https://smil.cmm.minesparis.psl.eu/images/lena.png")
   * im2 = sp.Image(im1)
   * kern = [ 0.0545, 0.2442, 0.4026, 0.2442, 0.0545 ]
   * sp.horizConvolve(im1, kern, im2)
   *
   * im1.show()
   * im2.show()
   *
   * @endcode
   */
  // Inplace safe
  template <class T>
  RES_T horizConvolve(const Image<T> &imIn, const vector<double> &kernel,
                      Image<T> &imOut)
  {
    CHECK_ALLOCATED(&imIn, &imOut);
    CHECK_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    typename ImDtTypes<T>::sliceType linesIn  = imIn.getLines();
    typename ImDtTypes<T>::sliceType linesOut = imOut.getLines();
    typename ImDtTypes<T>::lineType  lOut;

    int imW          = imIn.getWidth();
    int kernelRadius = (kernel.size() - 1) / 2;
    //         int kLen = 2*kernelRadius+1;

    double *partialKernWeights = new double[kernelRadius];
    double  pkwSum             = 0;
    for (int i = 0; i < kernelRadius; i++)
      pkwSum += kernel[i];
    for (int i = 0; i < kernelRadius; i++) {
      pkwSum += kernel[i + kernelRadius];
      partialKernWeights[i] = pkwSum;
    }

    typedef double      bufType; // If float, loops are vectorized
    BufferPool<bufType> bufferPool(imW);

#ifdef USE_OPEN_MP
    int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel private(lOut) num_threads(nthreads)
#endif // USE_OPEN_MP
    {
      typename ImDtTypes<bufType>::lineType lIn = bufferPool.getBuffer();
      double                                sum;

#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
      for (int y = 0; y < (int) imIn.getLineCount(); y++) {
        copyLine<T, bufType>(linesIn[y], imW, lIn);
        lOut = linesOut[y];

        // left pixels
        for (int x = 0; x < kernelRadius; x++) {
          sum = 0;
          for (int i = -x; i <= kernelRadius; i++)
            sum += kernel[i + kernelRadius] * lIn[x + i];
          lOut[x] = T(sum / partialKernWeights[x]);
        }

        // center pixels
        for (int x = kernelRadius; x < imW - kernelRadius; x++) {
          sum = 0;
          for (int i = -kernelRadius; i <= kernelRadius; i++)
            sum += kernel[i + kernelRadius] * lIn[x + i];
          lOut[x] = T(sum);
        }

        // right pixels
        for (int x = imW - kernelRadius; x < imW; x++) {
          sum = 0;
          for (int i = -kernelRadius; i < imW - x; i++)
            sum += kernel[i + kernelRadius] * lIn[x + i];
          lOut[x] = T(sum / partialKernWeights[imW - 1 - x]);
        }
      }
    }
    delete[] partialKernWeights;
    return RES_OK;
  }

  /**
   * vertConvolve() - 2D Vertical convolution
   *
   * 2D vertical convolution using a @TB{1D kernel}
   *
   * @param[in] imIn : input image
   * @param[in] kernel : an <b>1D kernel</b> (vector)
   * @param[out] imOut : output image
   *
   * @see horizConvolve()
   */
  template <class T>
  RES_T vertConvolve(const Image<T> &imIn, const vector<double> &kernel,
                     Image<T> &imOut)
  {
    CHECK_ALLOCATED(&imIn, &imOut);
    CHECK_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    typename ImDtTypes<T>::volType   slicesIn  = imIn.getSlices();
    typename ImDtTypes<T>::volType   slicesOut = imOut.getSlices();
    typename ImDtTypes<T>::sliceType sIn, sOut;

    int imW          = imIn.getWidth();
    int imH          = imIn.getHeight();
    int imD          = imIn.getDepth();
    int kernelRadius = (kernel.size() - 1) / 2;

    double *partialKernWeights = new double[kernelRadius];
    double  pkwSum             = 0;
    for (int i = 0; i < kernelRadius; i++)
      pkwSum += kernel[i];
    for (int i = 0; i < kernelRadius; i++) {
      pkwSum += kernel[i + kernelRadius];
      partialKernWeights[i] = pkwSum;
    }

    typedef double      bufType; // If double, loops are vectorized
    BufferPool<bufType> bufferPool(imW);

    for (int z = 0; z < imD; z++) {
#ifdef USE_OPEN_MP
      int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel private(sIn, sOut) num_threads(nthreads)
#endif // USE_OPEN_MP
      {
        sIn  = slicesIn[z];
        sOut = slicesOut[z];
        double sum;

#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
        for (int x = 0; x < imW; x++) {
          // Top pixels
          for (int y = 0; y < kernelRadius; y++) {
            sum = 0;
            for (int i = -y; i < kernelRadius + 1; i++)
              sum += kernel[i + kernelRadius] * sIn[y + i][x];
            sOut[y][x] = T(sum / partialKernWeights[y]);
          }

          // Center pixels
          for (int y = kernelRadius; y < imH - kernelRadius; y++) {
            sum = 0;
            for (int i = -kernelRadius; i <= kernelRadius; i++)
              sum += kernel[i + kernelRadius] * sIn[y + i][x];
            sOut[y][x] = T(sum);
          }

          // Bottom pixels
          for (int y = imH - kernelRadius; y < imH; y++) {
            sum = 0;
            for (int i = -kernelRadius; i < imH - y; i++)
              sum += kernel[i + kernelRadius] * sIn[y + i][x];
            sOut[y][x] = T(sum / partialKernWeights[imH - 1 - y]);
          }
        }
      }
    }
    delete[] partialKernWeights;
    return RES_OK;
  }

  /**
   * convolve() - 2D Convolution
   *
   * 2D convolution by & @TB{1D kernel}. Vertical convolution followed by
   * an horizontal convolution using the same @TB{1D kernel}.
   *
   * @param[in] imIn : input image
   * @param[in] kernel : the @TB{1D kernel} (vector)
   * @param[out] imOut : output image
   *
   * @see horizConvolve()
   */
  template <class T>
  RES_T convolve(const Image<T> &imIn, const vector<double> &kernel,
                 Image<T> &imOut)
  {
    if (&imIn == &imOut) {
      Image<T> tmpIm(imIn, true); // clone
      return convolve(tmpIm, kernel, imOut);
    }

    CHECK_ALLOCATED(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    vertConvolve(imIn, kernel, imOut);
    // inplace safe
    horizConvolve(imOut, kernel, imOut);

    return RES_OK;
  }

  /** @}*/

} // namespace smil

#endif // _D_IMAGE_DRAW_HPP
