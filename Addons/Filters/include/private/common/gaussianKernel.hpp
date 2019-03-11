/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2019, Centre de Morphologie Mathematique
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

#ifndef PI
#define PI 3.14159235
#endif


namespace smil
{

  template<typename T>
  inline T Max(T a, T b)
  {
    return (a >= b) ? a : b;
  }

  template<typename T>
  inline T Min(T a, T b)
  {
    return (a <= b) ? a : b;
  }

  /*
   *
   */
  class GaussianKernel
  {
  public:
#define ABS(i) ((i) >= 0 ? (i) : -(i))

    double getValue(int i, int j = 0, int k = 0)
    {
      double p;

      if (ABS(i) > radius || ABS(j) > radius || ABS(k) > radius)
        return 0.;

      p = kernel[ABS(i)];
      p *= kernel[ABS(j)];
      p *= kernel[ABS(k)];

      return p;
    }

    double getSum()
    {
      double sum = 0;

      for (int i = -radius; i <= radius; i++)
        for (int j = -radius; j <= radius; j++)
          for (int k = -radius; k <= radius; k++)
            sum += getValue(i, j, k);
      return sum;
    }

    GaussianKernel(double sigma, int radius) : radius(radius), sigma(sigma)
    {
      kernel = new double[radius + 1];
      for (int i = 0; i <= radius; i++) {
        kernel[i] = exp(-(i * i) / (sigma * sigma)) / sqrt(2 * PI * sigma);
      }
    }

    ~GaussianKernel()
    {
      delete[] kernel;
    }

    template <typename T> void Convolve(T *in, int W, int H, int, int D, T *out)
    {
      for (int z = 0; z < D; z++) {
        for (int y = 0; y < H; y++) {
          for (int x = 0; x < W; x++) {
            int loX    = MAX(0, x - radius);
            int hiX    = MIN(x + radius, W - 1);
            int loY    = MAX(0, y - radius);
            int hiY    = MIN(y + radius, H - 1);
            int loZ    = MAX(0, z - radius);
            int hiZ    = MIN(z + radius, D - 1);
            double sum = 0;
            for (int k = loZ; k <= hiZ; k++) {
              for (int j = loY; j <= hiY; j++) {
                for (int i = loX; i <= hiX; i++) {
                  sum += in[((z + k) * D + (y + j)) * W + x + i] *
                         getValue(i, j, k);
                }
              }
            }
            out[(z * D + y) * W + x] = sum / getSum();
          }
        }
      }
    }

  private:
    double *kernel;
    int radius;
    double sigma;
  };
  
  template<class T>
  RES_T ImGaussianFilter(Image<T> &imIn, double sigma, int radius, Image<T> imOut)
  {
    return RES_OK;
  }
} // namespace smil

#endif // _D_GAUSSIAN_KERNEL_HPP_
