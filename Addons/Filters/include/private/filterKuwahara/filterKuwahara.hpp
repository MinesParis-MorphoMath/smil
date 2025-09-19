#ifndef __KUWAHARA_FILTER_T_HPP__
#define __KUWAHARA_FILTER_T_HPP__

namespace smil
{
  template <class T>
  RES_T kuwaharaFilter(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    // MORPHEE_ENTER_FUNCTION("t_kuwaharaFilter");

    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    // TODO: check that image is 2D
    if (S[2] > 1) {
      // Error : this is a 3D image
      return RES_ERR;
    }

    int W, H;
    W = S[0];
    H = S[1];

    typename ImDtTypes<T>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T>::lineType bufferOut = imOut.getPixels();

    int diameter = radius * 2 + 1;
    int size2    = (diameter + 1) / 2;
    int offset   = (diameter - 1) / 2;

    int width2  = W + offset;
    int height2 = H + offset;

    double *mean     = new double[width2 * height2];
    double *variance = new double[width2 * height2];

    // Creation of the mean and variance map
    double sum, sum2;
    int    n, xbase, ybase;
    T      v = 0;
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

    int    xbase2 = 0, ybase2 = 0;
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
        bufferOut[x1 + y1 * W] = (T) (mean[xbase2 + ybase2 * width2]);
      }

    delete[] variance;
    delete[] mean;

    return RES_OK;
  }

  template <class T1, class T2>
  RES_T kuwaharaFilterRGB(const Image<T1> &imIn, const int radius,
                          Image<T2> &imOut)
  {
    return RES_OK;
  }

} // namespace smil
#endif
