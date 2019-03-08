#ifndef __FAST_SIGMA_FILTER_T_HPP__
#define __FAST_SIGMA_FILTER_T_HPP__

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/morpheeImage.hpp>

namespace morphee
{
  namespace filters
  {
    template <class T1, class T2>
    RES_C t_ImSigmaFilter(const Image<T1> &imIn, const UINT8 radius,
                          const double sigma, const double percentageNbMinPixel,
                          const bool excludeOutlier, Image<T2> &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImSigmaFilter");
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
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();

      int i, j, k, l;
      for (j = 0; j < H; j++)
        for (i = 0; i < W; i++) {
          double Mean = 0, Std = 0, lowerBound, higherBound;
          int NbPixel         = 0;
          int NbPixelInKernel = 0;
          double Sum = 0, Sum2 = 0;

          // Compute the mean and the std
          for (k = -radius; k <= radius; k++)
            if (i + k >= 0 && i + k < W)
              for (l = -radius; l <= radius; l++)
                if (j + l >= 0 && j + l < H) {
                  Sum += bufferIn[(i + k + (j + l) * W)];
                  Sum2 += bufferIn[(i + k + (j + l) * W)] *
                          bufferIn[(i + k + (j + l) * W)];
                  NbPixel++;
                }

          Mean = Sum / (double) NbPixel;
          Std  = sqrt((Sum2 / (double) NbPixel - (Mean * Mean)));

          lowerBound  = bufferIn[(i + j * W)] - (Std * sigma);
          higherBound = bufferIn[(i + j * W)] + (Std * sigma);

          Sum = 0;
          // Compute the kernel mean
          for (k = -radius; k <= radius; k++)
            if (i + k >= 0 && i + k < W)
              for (l = -radius; l <= radius; l++)
                if (j + l >= 0 && j + l < H &&
                    bufferIn[(i + k + (j + l) * W)] >= lowerBound &&
                    bufferIn[(i + k + (j + l) * W)] <= higherBound) {
                  Sum += bufferIn[(i + k + (j + l) * W)];
                  NbPixelInKernel++;
                }
          if (NbPixelInKernel <
              NbPixel * percentageNbMinPixel) { // On prend le vrai mean
            if (excludeOutlier)
              bufferOut[(i + j * W)] =
                  (T2)((Mean * NbPixel - bufferIn[(i + j * W)]) /
                       (double) (NbPixel - 1));
            else
              bufferOut[(i + j * W)] = (T2) Mean;
          } else
            bufferOut[(i + j * W)] = (T2)(Sum / (double) NbPixelInKernel);
        }

      return RES_OK;
    }

    template <class T1, class T2>
    RES_C t_ImSigmaFilterRGB(const Image<T1> &imIn, const UINT8 radius,
                             const double sigma,
                             const double percentageNbMinPixel,
                             const bool excludeOutlier, Image<T2> &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImSigmaFilterRGB");

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
      int i, j, k, l;
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();

      double *pixelsf = new double[W * H * 3];
      for (i = 0; i < W * H; i++) {
        int r = (int) bufferIn[i].channel1;
        int g = (int) bufferIn[i].channel2;
        int b = (int) bufferIn[i].channel3;

        pixelsf[i * 3]     = 0.299f * r + 0.587f * g + 0.114f * b;    // Y
        pixelsf[i * 3 + 1] = 0.5957f * r - 0.2744f * g - 0.3212f * b; // I
        pixelsf[i * 3 + 2] = 0.2114f * r - 0.5226f * g + 0.3111f * b; // Q
      }

      for (j = 0; j < H; j++)
        for (i = 0; i < W; i++) {
          double Mean = 0, Std = 0, lowerBound, higherBound;
          int NbPixel = 0;

          double Sum = 0, Sum2 = 0;

          // Compute the mean and the std
          for (k = -radius; k <= radius; k++)
            if (i + k >= 0 && i + k < W)
              for (l = -radius; l <= radius; l++)
                if (j + l >= 0 && j + l < H) {
                  Sum += pixelsf[(i + k + (j + l) * W) * 3];
                  Sum2 += pixelsf[(i + k + (j + l) * W) * 3] *
                          pixelsf[(i + k + (j + l) * W) * 3];
                  NbPixel++;
                }

          Mean = Sum / (double) NbPixel;
          Std  = sqrt((Sum2 - Sum * Sum / (double) NbPixel) / (double) NbPixel);

          lowerBound  = pixelsf[(i + j * W) * 3] - (Std * sigma);
          higherBound = pixelsf[(i + j * W) * 3] + (Std * sigma);

          int NbPixelInKernel = 0;
          Sum                 = 0;
          // Compute the kernel mean
          for (k = -radius; k <= radius; k++)
            if (i + k >= 0 && i + k < W)
              for (l = -radius; l <= radius; l++)
                if (j + l >= 0 && j + l < H &&
                    pixelsf[(i + k + (j + l) * W) * 3] >= lowerBound &&
                    pixelsf[(i + k + (j + l) * W) * 3] <= higherBound) {
                  Sum += pixelsf[(i + k + (j + l) * W) * 3];
                  NbPixelInKernel++;
                }
          double Out;
          if (NbPixelInKernel <
              NbPixel * percentageNbMinPixel) { // On prend le vrai mean
            if (excludeOutlier)
              Out = ((Mean * NbPixel - pixelsf[(i + j * W) * 3]) /
                     (double) (NbPixel - 1));
            else
              Out = Mean;
          } else
            Out = (Sum / (double) NbPixelInKernel);

          int r_ = (int) (Out + 0.9563f * pixelsf[(i + j * W) * 3 + 1] +
                          0.6210f * pixelsf[(i + j * W) * 3 + 2]);
          int g_ = (int) (Out - 0.2721f * pixelsf[(i + j * W) * 3 + 1] -
                          0.6473f * pixelsf[(i + j * W) * 3 + 2]);
          int b_ = (int) (Out - 1.1070f * pixelsf[(i + j * W) * 3 + 1] +
                          1.7046f * pixelsf[(i + j * W) * 3 + 2]);
          bufferOut[(i + j * W)].channel1 = r_;
          bufferOut[(i + j * W)].channel2 = g_;
          bufferOut[(i + j * W)].channel3 = b_;
        }

      delete[] pixelsf;
      return RES_OK;
    }

  } // namespace filters
} // namespace morphee
#endif