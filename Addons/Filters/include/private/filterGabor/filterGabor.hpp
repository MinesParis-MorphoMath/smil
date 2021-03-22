/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2021, Centre de Morphologie Mathematique
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
 *   2D Gabor filter implementation by Vincent Morard
 *
 * History :
 *   - XX/XX/XXXX - by Vincent Morard
 *     Just created it
 *   - 21/02/2019 - by Jose-Marcio Martins da Cruz
 *     Formatting and removing some warnings and minor differences
 *
 * __HEAD__ - Stop here !
 */

#ifndef __GABOR_FILTER_HPP__
#define __GABOR_FILTER_HPP__

#include <math.h>

namespace smil
{
#define VM_MAX(x, y) (((x) > (y)) ? (x) : (y))

  template <class T1, class T2>
  static RES_T _computeGaborFilterConvolution(T1 *bufferIn, int W, int H,
                                              double sigma, double theta,
                                              double lambda, double psi,
                                              double gamma, T2 *bufferOut)
  {
    int i, j, k, l;
    int Xmax, Ymax;
    int nstds = 3;
    int dx, dy;

    double dXmax, dYmax;

    //**********************
    // Generation of the kernel
    //**********************
    double sigma_x = sigma;
    double sigma_y = sigma * gamma;

    // Bounding box
    dXmax = VM_MAX(std::abs(nstds * sigma_x * cos(theta)),
                   std::abs(nstds * sigma_y * sin(theta)));
    dYmax = VM_MAX(std::abs(nstds * sigma_x * sin(theta)),
                   std::abs(nstds * sigma_y * cos(theta)));
    Xmax  = (int) VM_MAX(1, std::ceil(dXmax));
    Ymax  = (int) VM_MAX(1, std::ceil(dYmax));
    dx    = 2 * Xmax + 1;
    dy    = 2 * Ymax + 1;

    double *x_theta = new double[dx * dy];
    double *y_theta = new double[dx * dy];

    // 2D Rotation
    for (i = 0; i < dx; i++)
      for (j = 0; j < dy; j++) {
        x_theta[i + j * dx] =
            (i - dx / 2) * cos(theta) + (j - dy / 2) * sin(theta);
        y_theta[i + j * dx] =
            -(i - dx / 2) * sin(theta) + (j - dy / 2) * cos(theta);
      }

    double *gabor = new double[dx * dy];
    for (i = 0; i < dx; i++) {
      for (j = 0; j < dy; j++) {
        int pos = i + j * dx;

        gabor[pos] =
            std::exp(-0.5 * ((x_theta[pos] * x_theta[pos]) / (sigma_x * sigma_x) +
                        (y_theta[pos] * y_theta[pos]) / (sigma_y * sigma_y))) *
            cos(2 * 3.14159 / lambda * x_theta[pos] + psi);
      }
    }
    delete[] x_theta;
    delete[] y_theta;

    int I, J;
    double D;

    //****************************
    // Start of the convolution
    //****************************
    for (j = 0; j < H; j++)
      for (i = 0; i < W; i++) {
        D = 0;
        for (k = -dx / 2; k <= dx / 2; k++)
          for (l = -dy / 2; l <= dy / 2; l++) {
            I = (i + k) % W;
            J = (j + l) % H;
            if (I < 0)
              I += W; // Mirror
            if (J < 0)
              J += H;

            D += gabor[k + dx / 2 + (l + dx / 2) * dx] * bufferIn[I + J * W];
          }
        bufferOut[i + j * W] = (T2) D;
      }

    delete[] gabor;

    return RES_OK;
  }

  /*
   *
   */
  template <class T1>
  static void t_createGaborKernel(T1 *gabor, double sigma, double theta,
                                  double lambda, double psi, double gamma)
  {
    int i, j, Xmax, Ymax, nstds = 3, dx, dy;
    double dXmax, dYmax;

    //**********************
    // Generation of the kernel
    //**********************
    double sigma_x = sigma;
    double sigma_y = sigma * gamma;

    // Bounding box
    dXmax = VM_MAX(std::fabs(nstds * sigma_x * cos(theta)),
                   std::fabs(nstds * sigma_y * sin(theta)));
    dYmax = VM_MAX(std::fabs(nstds * sigma_x * sin(theta)),
                   std::fabs(nstds * sigma_y * cos(theta)));
    Xmax  = (int) VM_MAX(1, std::ceil(dXmax));
    Ymax  = (int) VM_MAX(1, std::ceil(dYmax));
    dx    = 2 * Xmax + 1;
    dy    = 2 * Ymax + 1;

    double *x_theta = new double[dx * dy];
    double *y_theta = new double[dx * dy];

    // 2D Rotation
    for (i = 0; i < dx; i++)
      for (j = 0; j < dy; j++) {
        x_theta[i + j * dx] =
            (i - dx / 2) * cos(theta) + (j - dy / 2) * sin(theta);
        y_theta[i + j * dx] =
            -(i - dx / 2) * sin(theta) + (j - dy / 2) * cos(theta);
      }

    for (j = 0; j < dy; j++)
      for (i = 0; i < dx; i++)
        gabor[i + j * dx] =
            (T1) std::exp(-0.5 * ((x_theta[i + j * dx] * x_theta[i + j * dx]) /
                                 (sigma_x * sigma_x) +
                             (y_theta[i + j * dx] * y_theta[i + j * dx]) /
                                 (sigma_y * sigma_y))) *
            cos(2 * 3.14159 / lambda * x_theta[i + j * dx] + psi);
    delete[] x_theta;
    delete[] y_theta;
  }

  /*
   *
   */
  template <class T>
  RES_T gaborFilterConvolution(const Image<T> &imIn, double sigma,
                                 double theta, double lambda, double psi,
                                 double gamma, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    if (S[2] > 1) {
      // This is a 3D Image...
      return RES_ERR;
    }

    typename ImDtTypes<T>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T>::lineType bufferOut = imOut.getPixels();

    _computeGaborFilterConvolution(bufferIn, S[0], S[1], sigma, theta, lambda,
                                   psi, gamma, bufferOut);

    return RES_OK;
  }

  /*
   *
   */
  template <class T>
  RES_T gaborFilterConvolutionNorm(const Image<T> &imIn, double sigma,
                                     double theta, double lambda, double psi,
                                     double gamma, double Min, double Max,
                                     Image<T> &imOut, Image<T> &imGabor)
  {
    ASSERT_ALLOCATED(&imIn, &imOut, &imGabor);
    ASSERT_SAME_SIZE(&imIn, &imOut, &imGabor);

    RES_T res = gaborFilterConvolution(imIn, sigma, theta, lambda, psi, gamma,
                                         imGabor);
    if (res != RES_OK) {
      // tell something
      return res;
    }

    size_t S[3];
    imIn.getSize(S);

    if (S[2] > 1) {
      // This is a 3D Image...
      return RES_ERR;
    }

    typename ImDtTypes<T>::lineType bufferGabor = imGabor.getPixels();
    typename ImDtTypes<T>::lineType bufferOut   = imOut.getPixels();

    int W, H;
    W = S[0];
    H = S[1];

    T dtMax = imOut.getDataTypeMax();
    for (int i = W * H - 1; i >= 0; i--) {
      bufferOut[i] = (T)((bufferGabor[i] - Min) / (Max - Min) * dtMax);
    }
    return RES_OK;
  }

  /*
   *
   */
  template <class T>
  RES_T gaborFilterConvolutionNormAuto(const Image<T> &imIn, double sigma,
                                         double theta, double lambda,
                                         double psi, double gamma, double *Min,
                                         double *Max, Image<T> &imOut,
                                         Image<T> &imGabor)
  {
    ASSERT_ALLOCATED(&imIn, &imOut, &imGabor);
    ASSERT_SAME_SIZE(&imIn, &imOut, &imGabor);

    RES_T res = gaborFilterConvolution(imIn, sigma, theta, lambda, psi, gamma,
                                         imGabor);
    if (res != RES_OK) {
      // tell something
      return res;
    }

    size_t S[3];
    imIn.getSize(S);

    if (S[2] > 1) {
      // This is a 3D Image...
      return RES_ERR;
    }

    typename ImDtTypes<T>::lineType bufferGabor = imGabor.getPixels();
    typename ImDtTypes<T>::lineType bufferOut   = imOut.getPixels();

    int W, H;
    W = S[0];
    H = S[1];

    *Min = bufferGabor[0];
    *Max = bufferGabor[0];
    for (int i = W * H - 1; i > 0; i--) {
      if (*Min > bufferGabor[i])
        *Min = bufferGabor[i];
      if (*Max < bufferGabor[i])
        *Max = bufferGabor[i];
    }

    T dtMax = imOut.getDataTypeMax();
    for (int i = W * H - 1; i >= 0; i--) {
      bufferOut[i] = (T)((bufferGabor[i] - *Min) / (*Max - *Min) * dtMax);
    }
    return RES_OK;
  }
} // namespace smil

#endif
