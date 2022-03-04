/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2022, Centre de Morphologie Mathematique
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
 *   This file does... some very complex morphological operation...
 *
 * History :
 *   - 09/03/2019 - by Jose-Marcio Martins da Cruz
 *     Just created it..
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_GAUSSIAN_KERNEL_HPP_
#define _D_GAUSSIAN_KERNEL_HPP_

namespace smil
{
  template <typename T> class GaussianFilterClass
  {
  private:
    vector<double> kernel;
    int radius;
    double sigma;

    void setupKernel()
    {
      kernel.resize(2 * radius + 1);

      double sum = 0;
      for (int i = -radius; i <= radius; i++) {
        kernel[i + radius] = exp(-(pow(i / sigma, 2) / 2));
        sum += kernel[i + radius];
      }
      for (int i = -radius; i <= radius; i++) {
        kernel[i + radius] /= sum;
      }
    }

    double getKernelValue(int i)
    {
      if ((i < -radius) || (i > radius))
        return 0.;

      return kernel[i + radius];
    }

  public:
    GaussianFilterClass(int radius, double sigma) : radius(radius), sigma(sigma)
    {
      setupKernel();
    }

    GaussianFilterClass(int radius = 2) : radius(radius)
    {
      sigma = radius / 2.;
      setupKernel();
    }

    ~GaussianFilterClass()
    {
    }

    RES_T Convolve(Image<T> &imIn, int radius, Image<T> &imOut)
    {
      this->radius = radius;
      this->sigma  = radius / 2.;
      setupKernel();

      return this->Convolve(imIn, imOut);
    }

    RES_T Convolve(Image<T> &imIn, Image<T> &imOut)
    {
      typename ImDtTypes<T>::lineType in  = imIn.getPixels();
      typename ImDtTypes<T>::lineType out = imOut.getPixels();

      off_t W = imIn.getWidth();
      off_t H = imIn.getHeight();
      off_t D = imIn.getDepth();

      size_t nbPixels = imIn.getPixelCount();

#ifdef USE_OPEN_MP
      int nthreads = Core::getInstance()->getNumberOfThreads();
#endif // USE_OPEN_MP

      /*
       * convolution in X
       */
      vector<T> outX(nbPixels, 0);
#ifdef USE_OPEN_MP
#pragma omp parallel num_threads(nthreads)
#endif // USE_OPEN_MP
      {
#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
        for (off_t z = 0; z < D; z++) {
          for (off_t y = 0; y < H; y++) {
            for (off_t x = 0; x < W; x++) {
              off_t i0    = (z * H + y) * W + x;
              double sumV = 0.;
              double sumK = 0.;

              for (off_t i = -radius; i <= radius; i++) {
                if ((x + i < 0) || (x + i > W - 1))
                  continue;
                double valK = getKernelValue(i);
                sumV += in[i0 + i] * valK;
                sumK += valK;
              }
              outX[i0] = (T)(sumV / sumK);
            }
          }
        }
      }

      /*
       * convolution in Y
       */
      vector<T> outY(nbPixels, 0);
#ifdef USE_OPEN_MP
#pragma omp parallel num_threads(nthreads)
#endif // USE_OPEN_MP
      {
        off_t stride = W;

#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
        for (off_t z = 0; z < D; z++) {
          for (off_t y = 0; y < H; y++) {
            for (off_t x = 0; x < W; x++) {
              off_t i0    = (z * H + y) * W + x;
              double sumV = 0.;
              double sumK = 0.;

              for (off_t i = -radius; i <= radius; i++) {
                if ((y + i < 0) || (y + i > H - 1))
                  continue;
                double valK = getKernelValue(i);
                sumV += outX[i0 + i * stride] * valK;
                sumK += valK;
              }
              outY[i0] = (T)(sumV / sumK);
            }
          }
        }
      }

      /*
       * convolution in Z
       */
#ifdef USE_OPEN_MP
#pragma omp parallel num_threads(nthreads)
#endif // USE_OPEN_MP
      {
        off_t stride = W * H;

#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
        for (off_t z = 0; z < D; z++) {
          for (off_t y = 0; y < H; y++) {
            for (off_t x = 0; x < W; x++) {
              off_t i0    = (z * H + y) * W + x;
              double sumV = 0.;
              double sumK = 0.;

              for (off_t i = -radius; i <= radius; i++) {
                if ((z + i < 0) || (z + i > D - 1))
                  continue;
                double valK = getKernelValue(i);
                sumV += outY[i0 + i * stride] * valK;
                sumK += valK;
              }
              out[i0] = (T)(sumV / sumK);
            }
          }
        }
      }
      return RES_OK;
    }
  };

  /*
   *
   */
  template <class T>
  RES_T imageGaussianFilter(Image<T> &imIn, int radius, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    GaussianFilterClass<T> k;

    ImageFreezer freeze(imOut);
    return k.Convolve(imIn, radius, imOut);
  }
} // namespace smil

#endif // _D_GAUSSIAN_KERNEL_HPP_
