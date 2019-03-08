#ifndef __KUWAHARA_FILTER_T_HPP__
#define __KUWAHARA_FILTER_T_HPP__

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/private/imageColorSpaceTransform_T.hpp>
#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/morpheeImage.hpp>

namespace morphee
{
  namespace filters
  {
    template <class T1, class T2>
    RES_C t_ImKuwaharaFilter(const Image<T1> &imIn, const int radius,
                             Image<T2> &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImKuwaharaFilter");

      // Check inputs
      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      if (imIn.getWzSize() > 1) {
        MORPHEE_REGISTER_ERROR("Only support 2D images");
        return RES_NOT_IMPLEMENTED;
      }

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();

      int diameter = radius * 2 + 1;
      int size2    = (diameter + 1) / 2;
      int offset   = (diameter - 1) / 2;

      int width2  = W + offset;
      int height2 = H + offset;

      double *mean     = new double[width2 * height2];
      double *variance = new double[width2 * height2];

      // Creation of the mean and variance map
      double sum, sum2;
      int n, xbase, ybase;
      T1 v = 0;
      for (int y1 = -offset; y1 < H; y1++) {
        for (int x1 = -offset; x1 < W; x1++) {
          sum  = 0;
          sum2 = 0;
          n    = 0;
          for (int x2 = x1; x2 < x1 + size2; x2++)
            if (x2 >= 0 && x2 < W)
              for (int y2 = y1; y2 < y1 + size2; y2++)
                if (y2 >= 0 && y2 < H) {
                  v = bufferIn[x2 + y2 * W];
                  sum += v;
                  sum2 += v * v;
                  n++;
                }
          mean[x1 + offset + (y1 + offset) * width2] =
              (double) (sum / (double) n);
          variance[x1 + offset + (y1 + offset) * width2] =
              (double) ((sum2 / (double) n) -
                        mean[x1 + offset + (y1 + offset) * width2] *
                            mean[x1 + offset + (y1 + offset) * width2]);
        }
      }

      int xbase2 = 0, ybase2 = 0;
      double var, Min;
      for (int y1 = 0; y1 < H; y1++)
        for (int x1 = 0; x1 < W; x1++) {
          Min   = 9999999;
          xbase = x1;
          ybase = y1;
          var   = variance[xbase + ybase * width2];
          if (var < Min) {
            Min    = var;
            xbase2 = xbase;
            ybase2 = ybase;
          }
          xbase = x1 + offset;
          var   = variance[xbase + ybase * width2];
          if (var < Min) {
            Min    = var;
            xbase2 = xbase;
            ybase2 = ybase;
          }
          ybase = y1 + offset;
          var   = variance[xbase + ybase * width2];
          if (var < Min) {
            Min    = var;
            xbase2 = xbase;
            ybase2 = ybase;
          }
          xbase = x1;
          var   = variance[xbase + ybase * width2];
          if (var < Min) {
            Min    = var;
            xbase2 = xbase;
            ybase2 = ybase;
          }
          bufferOut[x1 + y1 * W] = (T2)(mean[xbase2 + ybase2 * width2]);
        }

      delete[] variance;
      delete[] mean;
      return RES_OK;
    }

    template <class T1, class T2>
    RES_C t_ImKuwaharaFilterRGB(const Image<T1> &imIn, const int radius,
                                Image<T2> &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImKuwaharaFilterRGB");
      // Check inputs
      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      // Initialisation
      typedef typename T1::value_type Tin;
      morphee::Image<Tin> imInR = imIn.template t_getSame<Tin>();
      morphee::Image<Tin> imInG = imIn.template t_getSame<Tin>();
      morphee::Image<Tin> imInB = imIn.template t_getSame<Tin>();

      // Get the 3 channels
      RES_C res = t_colorSplitTo3(imIn, imInR, imInG, imInB);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR(
            "Error t_colorSplitTo3() in t_ImFastGaussianFilterRGB ");
        return res;
      }

      res = t_ImKuwaharaFilter(imInR, radius, imInR);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR(
            "t_ImKuwaharaFilter() in t_ImFastGaussianFilterRGB ");
        return res;
      }

      res = t_ImKuwaharaFilter(imInG, radius, imInG);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR(
            "t_ImKuwaharaFilter() in t_ImFastGaussianFilterRGB ");
        return res;
      }

      res = t_ImKuwaharaFilter(imInB, radius, imInB);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR(
            "t_ImKuwaharaFilter() in t_ImFastGaussianFilterRGB ");
        return res;
      }

      // Combinaision of the 3 channels
      res = t_colorComposeFrom3(imInR, imInG, imInB, imOut);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR(
            "Error colorComposeFrom3() in t_ImFastGaussianFilterRGB ");
        return res;
      }
      return RES_OK;
    }

  } // namespace filters
} // namespace morphee
#endif