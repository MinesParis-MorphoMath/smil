#ifndef __LINE_NAIVE_HPP__
#define __LINE_NAIVE_HPP__

#include "Core/include/DCore.h"

// Vincent Morard
// 11 octobre : naive implementation

namespace smil
{
  template <class T>
  RES_T ImLineDilateH(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn);
    ASSERT_ALLOCATED(&imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    int W, H, Z, i, j, k;
    W = imIn.getWidth();
    H = imIn.getHeight();
    Z = imIn.getDepth();
    if (Z > 0) {
      return RES_OK;
    }

    typename Image<T>::lineType bufferIn  = imIn.getPixels();
    typename Image<T>::lineType bufferOut = imOut.getPixels();
 
    T Max;

    // Dilation
    for (j = 0; j < H; j++)
      for (i = 0; i < W; i++) {
        Max = bufferIn[i + j * W];
        for (k = -radius; k <= radius; k++)
          if (i + k >= 0 && i + k < W)
            if (bufferIn[i + k + j * W] > Max)
              Max = bufferIn[i + k + j * W];
        bufferOut[i + j * W] = Max;
      }
    return RES_OK;
  }

  template <class T>
  RES_T ImLineErodeH(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn);
    ASSERT_ALLOCATED(&imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    int W, H, Z, i, j, k;
    W = imIn.getWidth();
    H = imIn.getHeight();
    Z = imIn.getDepth();
    if (Z > 1) {
      return RES_ERR;
    }

    typename Image<T>::lineType bufferIn  = imIn.getPixels();
    typename Image<T>::lineType bufferOut = imOut.getPixels();

    T Min;

    // Erosion
    for (j = 0; j < H; j++)
      for (i = 0; i < W; i++) {
        Min = bufferIn[i + j * W];
        for (k = -radius; k <= radius; k++)
          if (i + k >= 0 && i + k < W)
            if (bufferIn[i + k + j * W] < Min)
              Min = bufferIn[i + k + j * W];
        bufferOut[i + j * W] = Min;
      }
    return RES_OK;
  }

  template <class T>
  RES_T ImLineErodeV(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn);
    ASSERT_ALLOCATED(&imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    int W, H, Z, i, j, k;
    W = imIn.getWidth();
    H = imIn.getHeight();
    Z = imIn.getDepth();
    if (Z > 1) {
      return RES_ERR;
    }

    typename Image<T>::lineType bufferIn  = imIn.getPixels();
    typename Image<T>::lineType bufferOut = imOut.getPixels();

    T Min;

    // Erosion
    for (i = 0; i < W; i++)
      for (j = 0; j < H; j++) {
        Min = bufferIn[i + j * W];
        for (k = -radius; k <= radius; k++)
          if (j + k >= 0 && j + k < H)
            if (bufferIn[i + (j + k) * W] < Min)
              Min = bufferIn[i + (j + k) * W];
        bufferOut[i + j * W] = Min;
      }
    return RES_OK;
  }

  template <class T>
  RES_T ImLineDilateV(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn);
    ASSERT_ALLOCATED(&imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    int W, H, Z, i, j, k;
    W = imIn.getWidth();
    H = imIn.getHeight();
    Z = imIn.getDepth();
    if (Z > 0) {
      return RES_OK;
    }

    typename Image<T>::lineType bufferIn  = imIn.getPixels();
    typename Image<T>::lineType bufferOut = imOut.getPixels();

    T Max;

    // Dilation
    for (i = 0; i < W; i++)
      for (j = 0; j < H; j++) {
        Max = bufferIn[i + j * W];
        for (k = -radius; k <= radius; k++)
          if (j + k >= 0 && j + k < H)
            if (bufferIn[i + (j + k) * W] > Max)
              Max = bufferIn[i + (j + k) * W];
        bufferOut[i + j * W] = Max;
      }
    return RES_OK;
  }

  // Interface
  
  template <class T>
  RES_T ImLineOpen(const Image<T> &imIn, const int radius, bool horizontal,
                     Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineErode(imIn, radius, horizontal, imTmp);
    if (res == RES_OK)
      res = ImLineDilate(imTmp, radius, horizontal, imOut);
    return res;
  }

  template <class T>
  RES_T ImLineClose(const Image<T> &imIn, const int radius, bool horizontal,
                     Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineDilate(imIn, radius, horizontal, imTmp);
    if (res == RES_OK)
      res = ImLineErode(imTmp, radius, horizontal, imOut);
    return res;
  }

  template <class T>
  RES_T ImLineDilate(const Image<T> &imIn, const int radius, bool horizontal,
                     Image<T> &imOut)
  {
    if (horizontal)
      return ImLineDilateH(imIn, radius, imOut);
    return ImLineDilateV(imIn, radius, imOut);
  }

  template <class T>
  RES_T ImLineErode(const Image<T> &imIn, const int radius, bool horizontal,
                    Image<T> &imOut)
  {
    if (horizontal)
      return ImLineErodeH(imIn, radius, imOut);
    return ImLineErodeV(imIn, radius, imOut);
  }

} // namespace smil
#endif
