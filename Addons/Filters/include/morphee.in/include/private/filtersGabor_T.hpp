#ifndef __GABOR_FILTER_T_HPP__
#define __GABOR_FILTER_T_HPP__

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/morpheeImage.hpp>

namespace morphee
{
  namespace filters
  {
    /**
     * @author Vincent Morard
     * @brief Functions to compute Gabor filters by convolution
     * For the computation by FFT, we give a function createGaborFilter to
     * create a gabor filter
     */

#define VM_MAX(x, y) ((x > y) ? (x) : (y))

    template <class T1, class T2>
    RES_C ComputeGaborFilterConvolution(T1 *bufferIn, int W, int H,
                                        double sigma, double theta,
                                        double lambda, double psi, double gamma,
                                        Image<T2> &imOut)
    {
      int i, j, k, l, Xmax, Ymax, nstds = 3, dx, dy;
      double dXmax, dYmax;

      //**********************
      // Generation of the kernel
      //**********************
      double sigma_x = sigma;
      double sigma_y = sigma * gamma;

      // Bounding box
      dXmax = VM_MAX(abs(nstds * sigma_x * cos(theta)),
                     abs(nstds * sigma_y * sin(theta)));
      dYmax = VM_MAX(abs(nstds * sigma_x * sin(theta)),
                     abs(nstds * sigma_y * cos(theta)));
      Xmax  = (int) VM_MAX(1, ceil(dXmax));
      Ymax  = (int) VM_MAX(1, ceil(dYmax));
      dx    = 2 * Xmax + 1;
      dy    = 2 * Ymax + 1;

      double *x_theta = new double[dx * dy];
      double *y_theta = new double[dx * dy];
      if (x_theta == 0 || y_theta == 0) {
        return RES_ERROR_MEMORY;
      }

      // 2D Rotation
      for (i = 0; i < dx; i++) {
        for (j = 0; j < dy; j++) {
          x_theta[i + j * dx] =
              (i - dx / 2) * cos(theta) + (j - dy / 2) * sin(theta);
          y_theta[i + j * dx] =
              -(i - dx / 2) * sin(theta) + (j - dy / 2) * cos(theta);
        }
      }

      double *gabor = new double[dx * dy];
      if (gabor == 0) {
        delete[] x_theta;
        delete[] y_theta;
        return RES_ERROR_MEMORY;
      }

      for (i = 0; i < dx; i++) {
        for (j = 0; j < dy; j++)
          gabor[i + j * dx] =
              exp(-0.5 * ((x_theta[i + j * dx] * x_theta[i + j * dx]) /
                              (sigma_x * sigma_x) +
                          (y_theta[i + j * dx] * y_theta[i + j * dx]) /
                              (sigma_y * sigma_y))) *
              cos(2 * 3.14159 / lambda * x_theta[i + j * dx] + psi);
      }
      delete[] x_theta;
      delete[] y_theta;

      int I, J;
      double D;
      T2 *bufferOut = imOut.rawPointer();

      //****************************
      // Start of the convolution
      //****************************
      for (j = 0; j < H; j++) {
        for (i = 0; i < W; i++) {
          D = 0;
          for (k = -dx / 2; k <= dx / 2; k++) {
            for (l = -dy / 2; l <= dy / 2; l++) {
              I = ((i + k < 0 || i + k >= W) ? (i - k) : (i + k)); // Mirror
              J = ((j + l < 0 || j + l >= H) ? (j - l) : (j + l));
              D += gabor[k + dx / 2 + (l + dx / 2) * dx] * bufferIn[I + J * W];
            }
          }
          bufferOut[i + j * W] = (T2) D;
        }
      }
      delete[] gabor;
      return RES_OK;
    }

    template <class T1, class T2>
    RES_C t_ImGaborFilterConvolution(const Image<T1> &imIn, double sigma,
                                     double theta, double lambda, double psi,
                                     double gamma, Image<T2> &imOut)
    {
      RES_C res;

      // Check inputs
      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      int W, H, Z;

      W = imIn.getWxSize();
      H = imIn.getWySize();
      Z = imIn.getWzSize();

      if (Z != 1) {
        MORPHEE_REGISTER_ERROR("Image 3D not supported yet");
        return RES_ERROR;
      }

      res = ComputeGaborFilterConvolution(imIn.rawPointer(), W, H, sigma, theta,
                                          lambda, psi, gamma, imOut);

      return res;
    }

    template <class T1, class T2, class T3>
    RES_C t_ImGaborFilterConvolutionNorm(const Image<T1> &imIn, double sigma,
                                         double theta, double lambda,
                                         double psi, double gamma, double Min,
                                         double Max, Image<T2> &imOut,
                                         Image<T3> &imGabor)
    {
      int W, H;
      W = imIn.getWxSize();
      H = imIn.getWySize();

      RES_C res = t_ImGaborFilterConvolution(imIn, sigma, theta, lambda, psi,
                                             gamma, imGabor);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR("Error in t_ImGaborFilterConvolutionNorm ");
        return res;
      }

      T2 *bufferOut   = imOut.rawPointer();
      T3 *bufferGabor = imGabor.rawPointer();

      double Val;
      for (int i = W * H - 1; i >= 0; i--) {
        switch (imOut.getDataType()) {
        case sdtUINT8:
          Val = ((bufferGabor[i] - Min) / (Max - Min) * 255);
          if (Val > 255)
            Val = 255;
          if (Val < 0)
            Val = 0;
          break;
        case sdtUINT16:
          Val = ((bufferGabor[i] - Min) / (Max - Min) * 65535);
          if (Val > 65535)
            Val = 65535;
          if (Val < 0)
            Val = 0;
          break;
        case sdtUINT32:
          Val = ((bufferGabor[i] - Min) / (Max - Min) * 4294967295);
          if (Val > 4294967295)
            Val = 4294967295;
          if (Val < 0)
            Val = 0;
          break;
        }
        bufferOut[i] = (T2) Val;
      }

      return RES_OK;
    }

    template <class T1, class T2, class T3>
    RES_C
    t_ImGaborFilterConvolutionNormAuto(const Image<T1> &imIn, double sigma,
                                       double theta, double lambda, double psi,
                                       double gamma, double *Min, double *Max,
                                       Image<T2> &imOut, Image<T3> &imGabor)
    {
      RES_C res = t_ImGaborFilterConvolution(imIn, sigma, theta, lambda, psi,
                                             gamma, imGabor);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR("Error in t_ImGaborFilterConvolutionNormAuto ");
        return res;
      }

      int i, W = imIn.getWxSize();
      int H = imIn.getWySize();

      T2 *bufferOut   = imOut.rawPointer();
      T3 *bufferGabor = imGabor.rawPointer();

      *Min = bufferGabor[0];
      *Max = bufferGabor[0];
      for (i = W * H - 1; i > 0; i--) {
        if (*Min > bufferGabor[i])
          *Min = bufferGabor[i];
        if (*Max < bufferGabor[i])
          *Max = bufferGabor[i];
      }

      for (i = W * H - 1; i > 0; i--) {
        switch (imOut.getDataType()) {
        case sdtUINT8:
          bufferOut[i] = (T2)((bufferGabor[i] - *Min) / (*Max - (*Min)) * 255);
          break;
        case sdtUINT16:
          bufferOut[i] =
              (T2)((bufferGabor[i] - *Min) / (*Max - (*Min)) * 65535);
          break;
        case sdtUINT32:
          bufferOut[i] =
              (T2)((bufferGabor[i] - *Min) / (*Max - (*Min)) * 4294967295);
          break;
        }
      }

      return RES_OK;
    }

    // Creation of a Gabor kernel : used when we want a FFT treatment
    template <class T1>
    RES_C t_createGaborKernel(Image<T1> &imOut, double sigma, double theta,
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
      dXmax = VM_MAX(abs(nstds * sigma_x * cos(theta)),
                     abs(nstds * sigma_y * sin(theta)));
      dYmax = VM_MAX(abs(nstds * sigma_x * sin(theta)),
                     abs(nstds * sigma_y * cos(theta)));
      Xmax  = (int) VM_MAX(1, ceil(dXmax));
      Ymax  = (int) VM_MAX(1, ceil(dYmax));
      dx    = 2 * Xmax + 1;
      dy    = 2 * Ymax + 1;

      double *x_theta = new double[dx * dy];
      double *y_theta = new double[dx * dy];
      if (x_theta == 0 || y_theta == 0)
        return RES_ERROR_MEMORY;

      // 2D Rotation
      for (i = 0; i < dx; i++) {
        for (j = 0; j < dy; j++) {
          x_theta[i + j * dx] =
              (i - dx / 2) * cos(theta) + (j - dy / 2) * sin(theta);
          y_theta[i + j * dx] =
              -(i - dx / 2) * sin(theta) + (j - dy / 2) * cos(theta);
        }
      }

      T1 *gabor = imOut.rawPointer();
      for (j = 0; j < dy; j++)
        for (i = 0; i < dx; i++)
          gabor[i + j * dx] =
              (T1)(exp(-0.5 * ((x_theta[i + j * dx] * x_theta[i + j * dx]) /
                                   (sigma_x * sigma_x) +
                               (y_theta[i + j * dx] * y_theta[i + j * dx]) /
                                   (sigma_y * sigma_y))) *
                   cos(2 * 3.14159 / lambda * x_theta[i + j * dx] + psi));
      delete[] x_theta;
      delete[] y_theta;

      return RES_OK;
    }

    // Convert the double into a UINT8 kernel for visualisation purposes
    template <class T1, class T2>
    RES_C t_ImDisplayKernel(const Image<T1> &imIn, Image<T2> &imOut)
    {
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();
      int i, W, H;
      double Min, Max;
      Min = bufferIn[0];
      Max = bufferIn[0];

      W = imIn.getWxSize();
      H = imIn.getWySize();

      for (i = W * H - 1; i > 0; i--) {
        if (bufferIn[i] < Min)
          Min = bufferIn[i];
        if (bufferIn[i] > Max)
          Max = bufferIn[i];
      }

      for (i = W * H - 1; i >= 0; i--)
        bufferOut[i] = (T2)((bufferIn[i] - Min) / (Max - Min) * 255);

      return RES_OK;
    }

    template <class T1, class T2>
    RES_C t_ImNormalized(const Image<T1> &imIn, double Max, Image<T2> &imOut)
    {
      int W, H;
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      double OutValue;

      for (int i = 0; i < W * H; i++) {
        OutValue = bufferIn[i] / Max;
        if (OutValue > 255)
          OutValue = 255;
        bufferOut[i] = (T2)(OutValue + 0.5);
      }
      return RES_OK;
    }

  } // namespace filters
} // namespace morphee
#endif