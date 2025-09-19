
#ifndef __MEAN_SHIFT_FILTER_T_HPP__
#define __MEAN_SHIFT_FILTER_T_HPP__

namespace smil
{
  template <class T>
  RES_T meanShiftFilter(const Image<T> &imIn, const UINT8 radius,
                        const int tonalDistance, Image<T> &imOut)
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

    int W, H;
    W = S[0];
    H = S[1];

    int    i, j;
    double shift;

    // pour tous les pixels
    for (j = 0; j < H; j++)
      for (i = 0; i < W; i++) {
        int    xc = i;
        int    yc = j;
        int    xcOld, ycOld;
        double YcOld;
        double Yc = bufferIn[(i + j * W)];

        int iters = 0;
        do {
          xcOld = xc;
          ycOld = yc;
          YcOld = Yc;

          double mx  = 0;
          double my  = 0;
          double mY  = 0;
          int    num = 0;

          for (int ry = -radius; ry <= radius; ry++) {
            int y2 = yc + ry;
            if (y2 >= 0 && y2 < H) {
              for (int rx = -radius; rx <= radius; rx++) {
                int x2 = xc + rx;
                if (x2 >= 0 && x2 < W) {
                  if (ry * ry + rx * rx <= radius * radius) {
                    double Y2 = bufferIn[x2 + y2 * W];
                    double dY = Yc - Y2;

                    if (dY * dY <= tonalDistance * tonalDistance) {
                      mx += x2;
                      my += y2;
                      mY += Y2;
                      num++;
                    }
                  }
                }
              }
            }
          }

          double num_ = 1.0 / (double) num;
          Yc          = mY * num_;
          xc          = (int) (mx * num_ + 0.5);
          yc          = (int) (my * num_ + 0.5);
          int    dx   = xc - xcOld;
          int    dy   = yc - ycOld;
          double dY   = Yc - YcOld;

          shift = dx * dx + dy * dy + dY * dY;
          iters++;
        } while (shift > 3 && iters < 100);
        bufferOut[i + j * W] = (T) Yc;
      }

    return RES_OK;
  }

#if 1
  template <class T>
  RES_T meanShiftFilterRGB(const Image<T> &imIn, const UINT8 radius,
                           const int tonalDistance, Image<T> &imOut)
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

    int W, H;
    W = S[0];
    H = S[1];

#if 0
    double *pixelsf = new double[W * H * 3];
    for (int i = 0; i < W * H; i++) {
      int r = (int) bufferIn[i].channel1;
      int g = (int) bufferIn[i].channel2;
      int b = (int) bufferIn[i].channel3;

      pixelsf[i * 3]     = 0.299f * r + 0.587f * g + 0.114f * b;    // Y
      pixelsf[i * 3 + 1] = 0.5957f * r - 0.2744f * g - 0.3212f * b; // I
      pixelsf[i * 3 + 2] = 0.2114f * r - 0.5226f * g + 0.3111f * b; // Q
    }

    double shift = 0;
    int iters    = 0;
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int xc = x;
        int yc = y;
        int xcOld, ycOld;
        double YcOld, IcOld, QcOld;
        int pos = y * W + x;

        double Yc = pixelsf[pos * 3];
        double Ic = pixelsf[pos * 3 + 1];
        double Qc = pixelsf[pos * 3 + 2];

        iters = 0;

        do {
          xcOld = xc;
          ycOld = yc;
          YcOld = Yc;
          IcOld = Ic;
          QcOld = Qc;

          double mx = 0;
          double my = 0;
          double mY = 0;
          double mI = 0;
          double mQ = 0;
          int num   = 0;

          for (int ry = -radius; ry <= radius; ry++) {
            int y2 = yc + ry;
            if (y2 >= 0 && y2 < H) {
              for (int rx = -radius; rx <= radius; rx++) {
                int x2 = xc + rx;
                if (x2 >= 0 && x2 < W) {
                  if (ry * ry + rx * rx <= radius * radius) {
                    double Y2 = pixelsf[(y2 * W + x2) * 3];
                    double I2 = pixelsf[(y2 * W + x2) * 3 + 1];
                    double Q2 = pixelsf[(y2 * W + x2) * 3 + 2];

                    double dY = Yc - Y2;
                    double dI = Ic - I2;
                    double dQ = Qc - Q2;

                    if (dY * dY + dI * dI + dQ * dQ <=
                        tonalDistance * tonalDistance) {
                      mx += x2;
                      my += y2;
                      mY += Y2;
                      mI += I2;
                      mQ += Q2;
                      num++;
                    }
                  }
                }
              }
            }
          }
          double num_ = 1.0 / (double) num;
          Yc          = mY * num_;
          Ic          = mI * num_;
          Qc          = mQ * num_;
          xc          = (int) (mx * num_ + 0.5);
          yc          = (int) (my * num_ + 0.5);
          int dx      = xc - xcOld;
          int dy      = yc - ycOld;
          double dY   = Yc - YcOld;
          double dI   = Ic - IcOld;
          double dQ   = Qc - QcOld;

          shift = dx * dx + dy * dy + dY * dY + dI * dI + dQ * dQ;
          iters++;
        } while (shift > 3 && iters < 100);

        int r_ = (int) (Yc + 0.9563f * Ic + 0.6210f * Qc);
        int g_ = (int) (Yc - 0.2721f * Ic - 0.6473f * Qc);
        int b_ = (int) (Yc - 1.1070f * Ic + 1.7046f * Qc);

        bufferOut[pos].channel1 = r_;
        bufferOut[pos].channel2 = g_;
        bufferOut[pos].channel3 = b_;
      }
    }

    delete[] pixelsf;
#endif
    return RES_OK;
  }
#endif

} // namespace smil
#endif
