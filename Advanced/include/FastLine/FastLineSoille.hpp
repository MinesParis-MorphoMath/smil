#ifndef __FAST_LINE_SOILLE_T_HPP__
#define __FAST_LINE_SOILLE_T_HPP__

#include "Core/include/DCore.h"

// Morph-M interface by Vincent Morard
// 1 september 2010

// February 23, 2006  Erik R. Urbach
// Email: erik@cs.rug.nl
// Implementation of algorithm by Soille et al. [1] for erosions and
// dilations with linear structuring elements (S.E.) at arbitrary angles.
// S.E. line drawing using Bresenham's Line Algorithm [2].
// Compilation: gcc -ansi -pedantic -Wall -O3 -o polygonsoille polygonsoille.c
// -lm
//
// Related papers:
// [1] P. Soille and E. Breen and R. Jones.
//     Recursive implementation of erosions and dilations along discrete
//     lines at arbitrary angles.
//     IEEE Transactions on Pattern Analysis and Machine Intelligence,
//     Vol. 18, Number 5, Pages 562-567, May 1996.
// [2] Donald Hearn and M. Pauline Baker
//     Computer Graphics, second edition
//     Prentice Hall

namespace smil
{
  // #define PI 3.14159265358979323846

  typedef unsigned char ubyte;
  typedef unsigned int uint;
  typedef unsigned long ulong;

  typedef struct SoilleImage SoilleImage;

  struct SoilleImage {
    ulong Width;
    ulong Height;
    void *Pixmap;
  };

  inline ulong ComputeLinePosDiag(long x, long y, ulong width, ulong height,
                                  ulong *p)
  {
    ulong idx;
    long x0;

    if (x < 0) {
      y -= x;
      x = 0;
    }
    if (y < 0) {
      x -= y;
      y = 0;
    }
    x0  = x;
    idx = y * width + x;
    for (; (x < (long) width) && (y < (long) height); x++) {
      *p = idx;
      idx += width + 1;
      p++;
      y++;
    }
    return (x - x0);
  } /* ComputeLinePosDiag */

  inline ulong ComputeLineNegDiag(long x, long y, ulong width, ulong height,
                                  ulong *p)
  {
    ulong idx;
    long x0;

    if (y >= (long) height) {
      x += y - height + 1;
      y = height - 1;
    }
    if (x >= (long) width)
      return (0);
    x0  = x;
    idx = y * width + x;
    *p  = idx;
    while ((x < (long) (width - 1)) && (y > 0)) {
      p++;
      x++;
      y--;
      idx -= width - 1;
      *p = idx;
    }
    return (x - x0 + 1);
  } /* ComputeLineNegDiag */

  inline ulong ComputeBresenhamLinePX(long x, long y, long dx, long dy,
                                      ulong phase, ulong width, ulong height,
                                      ulong *p)
  /* Computes pixel coords along line with slope 0<m<1 */
  /* Returns # of pixel coords (num) written to array p (num <= width) */
  {
    ulong idx;
    long x0;
    long dp = 2 * dy - 2 * phase, twody = 2 * dy, twodydx = 2 * dy - 2 * dx;

    while ((x < 0) || (y < 0)) {
      if (dp >= 0) {
        y++;
        dp += twodydx;
      } else
        dp += twody;
      x++;
    }
    x0  = x;
    idx = y * width + x;
    while ((x < (long) width) && (y < (long) height)) {
      *p = idx;
      p++;
      if (dp >= 0) {
        y++;
        idx += width;
        dp += twodydx;
      } else
        dp += twody;
      x++;
      idx++;
    }
    return (x - x0);
  } /* ComputeBresenhamLinePX */

  inline ulong ComputeBresenhamLineNX(long x, long y, long dx, long dy,
                                      ulong phase, ulong width, ulong height,
                                      ulong *p)
  /* Computes pixel coords along line with slope -1<m<0 */
  /* Returns # of pixel coords (num) written to array p (num <= width) */
  {
    ulong x0 = x, idx = y * width + x;
    long dp = 2 * dy - 2 * phase, twody = 2 * dy, twodydx = 2 * dy - 2 * dx;

    while (y >= (long) height) {
      if (dp >= 0) {
        y--;
        dp += twodydx;
      } else
        dp += twody;
      x++;
    }
    if (x >= (long) width)
      return (0);
    x0  = x;
    idx = y * width + x;
    *p  = idx;
    p++;
    while ((x < (long) width - 1) && (y > 0)) {
      if (dp >= 0) {
        y--;
        idx -= width;
        dp += twodydx;
      } else
        dp += twody;
      x++;
      idx++;
      *p = idx;
      p++;
    }
    return (x - x0 + 1);
  } /* ComputeBresenhamLineNX */

  inline ulong ComputeBresenhamLinePY(long x, long y, long dx, long dy,
                                      ulong phase, ulong width, ulong height,
                                      ulong *p)
  /* Computes pixel coords along line with slope m>1 */
  /* Returns # of pixel coords (num) written to array p (num <= height) */
  {
    ulong y0, idx;
    long dp = 2 * dx - 2 * phase, twodx = 2 * dx, twodxdy = 2 * dx - 2 * dy;

    while ((x < 0) || (y < 0)) {
      if (dp >= 0) {
        x++;
        dp += twodxdy;
      } else
        dp += twodx;
      y++;
    }
    y0  = y;
    idx = y * width + x;
    while ((y < (long) height) && (x < (long) width)) {
      *p = idx;
      p++;
      if (dp >= 0) {
        x++;
        idx++;
        dp += twodxdy;
      } else
        dp += twodx;
      y++;
      idx += width;
    }
    return (y - y0);
  } /* ComputeBresenhamLinePY */

  inline ulong ComputeBresenhamLineNY(long x, long y, long dx, long dy,
                                      ulong phase, ulong width, ulong height,
                                      ulong *p)
  /* Computes pixel coords along line with slope m<-1 */
  /* Returns # of pixel coords (num) written to array p (num <= height) */
  {
    ulong y0, idx;
    long dp = 2 * dx - 2 * phase, twodx = 2 * dx, twodxdy = 2 * dx - 2 * dy;

    while (x >= (long) width) {
      if (dp >= 0) {
        x--;
        dp += twodxdy;
      } else
        dp += twodx;
      y++;
    }
    if (y >= (long) height)
      return (0);
    y0  = y;
    idx = y * width + x;
    *p  = idx;
    p++;
    while ((y < (long) height - 1) && (x > 0) && (x < (long) width)) {
      if (dp >= 0) {
        x--;
        idx--;
        dp += twodxdy;
      } else
        dp += twodx;
      y++;
      idx += width;
      *p = idx;
      p++;
    }
    return (y - y0 + 1);
  } /* ComputeBresenhamLineNY */

  template <class T>
  inline void DilateHorLine(T *f, ulong width, ulong k, T *g, T *h, T *h2, T *r)
  /* k is length of SE in number of pixels */
  /* width is length of g, h, h2, and r */
  {
    ulong x, x2;

    for (x = 0; x < width; x++) {
      x2 = width - 1 - x;
      if (x % k) {
        g[x] = MAX(g[x - 1], f[x]);
      } else {
        g[x] = f[x];
      }

      if (((x2 % k) == (k - 1)) || (x2 == (width - 1))) {
        h[x2] = f[x2];
      } else {
        h[x2] = MAX(h[x2 + 1], f[x2]);
      }
    }
    if ((k == 1) || (width == 1))
      h[0] = f[0];
    else
      h[0] = MAX(h[1], f[0]);
    h2[width - 1] = f[width - 1];
    for (x = width - 2; x >= (width - k); x--)
      h2[x] = MAX(h2[x + 1], f[x]);
    h2[0] = MAX(h2[1], f[0]);

    if (width <= (k / 2)) {
      for (x = 0; x < width; x++)
        r[x] = g[width - 1];
    } else if (width <= k) {
      for (x = 0; x < (width - k / 2); x++)
        r[x] = g[x + k / 2];
      for (; x <= (k / 2); x++)
        r[x] = g[width - 1];
      for (; x < width; x++)
        r[x] = h[x - k / 2];
    } else /* width > k */
    {
      for (x = 0; x < (width - k / 2); x++) {
        if (x < (k / 2))
          r[x] = g[x + k / 2];
        else
          r[x] = MAX(g[x + k / 2], h[x - k / 2]);
      }
      for (x = width - k / 2; x < width; x++)
        r[x] = h2[x - k / 2];
    }
  } /* DilateHorLine */

  template <class T>
  inline void DilateVerLine(T *f, ulong width, ulong height, ulong k, T *g,
                            T *h, T *h2, T *r)
  /* k is length of SE in number of pixels */
  /* height is length of g, h, h2, and r */
  {
    ulong y, y2;

    for (y = 0; y < height; y++) {
      y2 = height - 1 - y;
      if (y % k) {
        g[y] = MAX(g[y - 1], f[y * width]);
      } else
        g[y] = f[y * width];

      if (((y2 % k) == (k - 1)) || (y2 == (height - 1)))
        h[y2] = f[y2 * width];
      else
        h[y2] = MAX(h[y2 + 1], f[y2 * width]);
    }

    if ((k == 1) || (height == 1))
      h[0] = f[0];
    else
      h[0] = MAX(h[1], f[0]);
    h2[height - 1] = f[(height - 1) * width];
    for (y = height - 2; y >= (height - k); y--)
      h2[y] = MAX(h2[y + 1], f[y * width]);
    h2[0] = MAX(h[1], f[0]);

    if (height <= (k / 2)) {
      for (y = 0; y < height; y++)
        r[y * width] = g[height - 1];
    } else if (height <= k) {
      for (y = 0; y < (height - k / 2); y++)
        r[y * width] = g[y + k / 2];
      for (; y <= (k / 2); y++)
        r[y * width] = g[height - 1];
      for (; y < height; y++)
        r[y * width] = h[y - k / 2];
    } else /* height > k */
    {
      for (y = 0; y < (height - k / 2); y++) {
        if (y < (k / 2))
          r[y * width] = g[y + k / 2];
        else
          r[y * width] = MAX(g[y + k / 2], h[y - k / 2]);
      }
      for (y = height - k / 2; y < height; y++)
        r[y * width] = h2[y - k / 2];
    }
  } /* DilateVerLine */

  template <class T>
  inline void DilateLine(T *f, SMIL_UNUSED ulong width,
                         SMIL_UNUSED ulong height, ulong k, ulong nx, ulong *p,
                         T *g, T *h, T *h2, T *r)
  /* k is length of SE in number of pixels */
  /* nx is length of p, g, h, and r */
  {
    ulong x, x2;

    for (x = 0; x < nx; x++) {
      x2 = nx - 1 - x;
      if (x % k) {
        g[x] = MAX(g[x - 1], f[p[x]]);
      } else
        g[x] = f[p[x]];

      if (((x2 % k) == (k - 1)) || (x2 == (nx - 1)))
        h[x2] = f[p[x2]];
      else
        h[x2] = MAX(h[x2 + 1], f[p[x2]]);
    }

    if ((k == 1) || (nx == 1))
      h[0] = f[p[0]];
    else
      h[0] = MAX(h[1], f[p[0]]);
    h2[nx - 1] = f[p[nx - 1]];
    if (nx >= 2) {
      for (x = nx - 2; (x > 0) && (x >= (nx - k)); x--)
        h2[x] = MAX(h2[x + 1], f[p[x]]);
      h2[0] = MAX(h2[1], f[p[0]]);
    }

    if (nx <= (k / 2)) {
      for (x = 0; x < nx; x++)
        r[p[x]] = g[nx - 1];
    } else if (nx <= k) {
      for (x = 0; x < (nx - k / 2); x++)
        r[p[x]] = g[x + k / 2];
      for (; x <= (k / 2); x++)
        r[p[x]] = g[nx - 1];
      for (; x < nx; x++)
        r[p[x]] = h[x - k / 2];
    } else /* nx > k */
    {
      for (x = 0; x < (nx - k / 2); x++) {
        if (x < (k / 2))
          r[p[x]] = g[x + k / 2];
        else
          r[p[x]] = MAX(g[x + k / 2], h[x - k / 2]);
      }
      for (x = nx - k / 2; x < nx; x++)
        r[p[x]] = h2[x - k / 2];
    }
  } /* DilateLine */

  template <class T>
  void ImageGrayDilateHor(SoilleImage *img, ulong k, T *g, T *h, T *h2,
                          SoilleImage *out)
  {
    T *f        = (T *) img->Pixmap;
    T *r        = (T *) out->Pixmap;
    ulong width = img->Width, y;

    for (y = 0; y < img->Height; y++) {
      DilateHorLine(f, width, k, g, h, h2, r);
      f += width;
      r += width;
    }
  } /* ImageGrayDilateHor */

  template <class T>
  void ImageGrayDilateVer(SoilleImage *img, ulong k, T *g, T *h, T *h2,
                          SoilleImage *out)
  {
    T *f        = (T *) img->Pixmap;
    T *r        = (T *) out->Pixmap;
    ulong width = img->Width, height = img->Height, x;

    for (x = 0; x < width; x++) {
      DilateVerLine(f, width, height, k, g, h, h2, r);
      f++;
      r++;
    }
  } /* ImageGrayDilateVer */

  template <class T>
  void ImageGrayDilateLine(SoilleImage *img, ulong k, long dx, long dy,
                           ulong phase, ulong *p, T *g, T *h, T *h2,
                           SoilleImage *out)
  {
    T *f        = (T *) img->Pixmap;
    T *r        = (T *) out->Pixmap;
    ulong width = img->Width, height = img->Height, nx;
    long x, y;

    if (dy == 0)
      ImageGrayDilateHor(img, k, g, h, h2, out);
    else if (dx == 0)
      ImageGrayDilateVer(img, k, g, h, h2, out);
    else if (abs(dx) == abs(dy)) {
      if (dx == -dy) {
        y  = 0;
        nx = ComputeLineNegDiag(0, y, width, height, p);
        while (nx > 0) {
          DilateLine(f, width, height, k, nx, p, g, h, h2, r);
          y++;
          nx = ComputeLineNegDiag(0, y, width, height, p);
        }
      } else {
        y  = height - 2;
        nx = ComputeLinePosDiag(0, y, width, height, p);
        while (nx > 0) {
          DilateLine(f, width, height, k, nx, p, g, h, h2, r);
          y--;
          nx = ComputeLinePosDiag(0, y, width, height, p);
        }
      }
    } else if (abs(dx) > abs(dy)) {
      if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
        dx = abs(dx);
        dy = abs(dy);
        y  = height - 1;
        nx = ComputeBresenhamLinePX(0, y, dx, dy, phase, width, height, p);
        while (nx > 0) {
          DilateLine(f, width, height, k, nx, p, g, h, h2, r);
          y--;
          nx = ComputeBresenhamLinePX(0, y, dx, dy, phase, width, height, p);
        }
      } else {
        dx = abs(dx);
        dy = abs(dy);
        y  = 0;
        nx = ComputeBresenhamLineNX(0, y, dx, dy, phase, width, height, p);
        while (nx > 0) {
          DilateLine(f, width, height, k, nx, p, g, h, h2, r);
          y++;
          nx = ComputeBresenhamLineNX(0, y, dx, dy, phase, width, height, p);
        }
      }
    } else {
      if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
        dx = abs(dx);
        dy = abs(dy);
        x  = width - 1;
        nx = ComputeBresenhamLinePY(x, 0, dx, dy, phase, width, height, p);
        while (nx > 0) {
          DilateLine(f, width, height, k, nx, p, g, h, h2, r);
          x--;
          nx = ComputeBresenhamLinePY(x, 0, dx, dy, phase, width, height, p);
        }
      } else {
        dx = abs(dx);
        dy = abs(dy);
        x  = 0;
        nx = ComputeBresenhamLineNY(x, 0, dx, dy, phase, width, height, p);
        while (nx > 0) {
          DilateLine(f, width, height, k, nx, p, g, h, h2, r);
          x++;
          nx = ComputeBresenhamLineNY(x, 0, dx, dy, phase, width, height, p);
        }
      }
    }
  } /* ImageGrayDilateLine */

  template <class T>
  RES_T ImLineDilate_Soille(const Image<T> &imIn, const int angle,
                            const int radius, Image<T> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn);
    ASSERT_ALLOCATED(&imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    int W, H;
    W = imIn.getWidth();
    H = imIn.getHeight();

    int maxnx = MAX(W, H);
    T *g      = new T[maxnx];
    T *h      = new T[maxnx];
    T *h2     = new T[maxnx];
    ulong *p  = new ulong[maxnx];

    SoilleImage MyImgIn, MyImgOut;

    MyImgIn.Width  = W;
    MyImgIn.Height = H;
    MyImgIn.Pixmap = (void *) imIn.getPixels();

    MyImgOut.Width  = W;
    MyImgOut.Height = H;
    MyImgOut.Pixmap = (void *) imOut.getPixels();

    int dx = (int) (cos(angle * PI / 180.0) * maxnx);
    int dy = (int) (-sin(angle * PI / 180.0) * maxnx);

    ImageGrayDilateLine(&MyImgIn, radius * 2 + 1, dx, dy, 1, p, g, h, h2,
                        &MyImgOut);

    delete[] g;
    delete[] h;
    delete[] h2;
    delete[] p;

    return RES_OK;
  }

  template <class T>
  RES_T ImLineErode_Soille(const Image<T> &imIn, const int angle,
                           const int radius, Image<T> &imOut)
  {
    RES_T r = RES_OK;
    Image<T> imTmp(imIn);

    r = inv(imIn, imTmp);
    if (r == RES_OK)
      r = smil::ImLineDilate_Soille(imTmp, angle, radius, imOut);
    if (r == RES_OK)
      r = inv(imOut, imOut);
    return r;
  }

  template <class T>
  RES_T ImLineOpen_Soille(const Image<T> &imIn, const int angle,
                          const int radius, Image<T> &imOut)
  {
    RES_T r = RES_OK;
    Image<T> imTmp(imIn);

    r = smil::ImLineErode_Soille(imIn, angle, radius, imTmp);
    if (r == RES_OK)
      r = smil::ImLineDilate_Soille(imTmp, angle, radius, imOut);
    return r;
  }

  template <class T>
  RES_T ImLineClose_Soille(const Image<T> &imIn, const int angle,
                           const int radius, Image<T> &imOut)
  {
    RES_T r = RES_OK;
    Image<T> imTmp(imIn);

    r = smil::ImLineDilate_Soille(imIn, angle, radius, imTmp);
    if (r == RES_OK)
      r = smil::ImLineErode_Soille(imTmp, angle, radius, imOut);
    return r;
  }

  template <class T>
  RES_T ImCircleDilate_Soille(const Image<T> &imIn, const int radius,
                              Image<T> &imOut)
  {
    RES_T r;
    int i, nbAngle = 8;
    double alpha, kalpha;

    alpha  = 0.;
    kalpha = (radius * PI / nbAngle * 0.5) *
                 std::max(fabs(cos(alpha)), fabs(sin(alpha))) +
             0.5;
    r = ImLineDilate_Soille(imIn, (int) alpha, (int) kalpha, imOut);
    if (r != RES_OK)
      return r;
    for (i = 1; i < nbAngle; i++) {
      alpha  = i * 180. / nbAngle;
      kalpha = (radius * PI / nbAngle * 0.5) *
                   std::max(fabs(cos(alpha)), fabs(sin(alpha))) +
               0.5;

      r = ImLineDilate_Soille(imOut, (int) alpha, (int) kalpha, imOut);
      if (r != RES_OK)
        break;
    }
    return r;
  }

  template <class T>
  RES_T ImCircleErode_Soille(const Image<T> &imIn, const int radius,
                             Image<T> &imOut)
  {
    RES_T r;
    int i, nbAngle = 8;
    double alpha, kalpha;

    alpha  = 0.;
    kalpha = (radius * PI / nbAngle * 0.5) *
                 std::max(fabs(cos(alpha)), fabs(sin(alpha))) +
             0.5;
    r = ImLineErode_Soille(imIn, (int) alpha, (int) kalpha, imOut);
    if (r != RES_OK)
      return r;
    for (i = 1; i < nbAngle; i++) {
      alpha  = i * 180. / nbAngle;
      kalpha = (radius * PI / nbAngle * 0.5) *
                   std::max(fabs(cos(alpha)), fabs(sin(alpha))) +
               0.5;

      r = ImLineErode_Soille(imIn, (int) alpha, (int) kalpha, imOut);
      if (r != RES_OK)
        break;
    }
    return r;
  }

  template <class T>
  RES_T ImCircleOpen_Soille(const Image<T> &imIn, const int radius,
                            Image<T> &imOut)
  {
    RES_T r;

    r = ImCircleErode_Soille(imIn, radius, imOut);
    if (r == RES_OK)
      r = ImCircleDilate_Soille(imOut, radius, imOut);
    return r;
  }

  template <class T>
  RES_T ImCircleClose_Soille(const Image<T> &imIn, const int radius,
                             Image<T> &imOut)
  {
    RES_T r;
    r = ImCircleDilate_Soille(imIn, radius, imOut);
    if (r == RES_OK)
      r = ImCircleErode_Soille(imOut, radius, imOut);
    return r;
  }

  template <class T>
  RES_T ImSquareDilate_Soille(const Image<T> &imIn, const int radius,
                              Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineDilate_Soille(imIn, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineDilate_Soille(imTmp, 90, radius, imOut);
    return res;
  }

  template <class T>
  RES_T ImSquareErode_Soille(const Image<T> &imIn, const int radius,
                             Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineErode_Soille(imIn, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineErode_Soille(imTmp, 90, radius, imOut);
    return res;
  }

  template <class T>
  RES_T ImSquareOpen_Soille(const Image<T> &imIn, const int radius,
                            Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineErode_Soille(imIn, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineErode_Soille(imTmp, 90, radius, imOut);
    if (res == RES_OK)
      res = ImLineDilate_Soille(imOut, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineDilate_Soille(imTmp, 90, radius, imOut);
    return res;
  }

  template <class T>
  RES_T ImSquareClose_Soille(const Image<T> &imIn, const int radius,
                             Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineDilate_Soille(imIn, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineDilate_Soille(imTmp, 90, radius, imOut);
    if (res == RES_OK)
      res = ImLineErode_Soille(imOut, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineErode_Soille(imTmp, 90, radius, imOut);
    return res;
  }
} // namespace smil

#endif
