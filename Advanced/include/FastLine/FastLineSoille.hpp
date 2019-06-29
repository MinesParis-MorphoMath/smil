#ifndef __FAST_LINE_SOILLE_T_HPP__
#define __FAST_LINE_SOILLE_T_HPP__

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/morpheeImage.hpp>

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

namespace morphee
{
  namespace FastLine
  {
#define PI 3.14159265358979323846

#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

    typedef unsigned char ubyte;
    typedef unsigned int uint;
    typedef unsigned long ulong;

    typedef struct ImageGray ImageGray;

    struct ImageGray {
      ulong Width;
      ulong Height;
      void *Pixmap;
    };

    ulong ComputeLinePosDiag(long x, long y, ulong width, ulong height,
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
      for (; (x < width) && (y < height); x++) {
        *p = idx;
        idx += width + 1;
        p++;
        y++;
      }
      return (x - x0);
    } /* ComputeLinePosDiag */

    ulong ComputeLineNegDiag(ulong x, ulong y, ulong width, ulong height,
                             ulong *p)
    {
      ulong idx;
      long x0;

      if (y >= height) {
        x += y - height + 1;
        y = height - 1;
      }
      if (x >= width)
        return (0);
      x0  = x;
      idx = y * width + x;
      *p  = idx;
      while ((x < width - 1) && (y > 0)) {
        p++;
        x++;
        y--;
        idx -= width - 1;
        *p = idx;
      }
      return (x - x0 + 1);
    } /* ComputeLineNegDiag */

    ulong ComputeBresenhamLinePX(long x, long y, long dx, long dy, ulong phase,
                                 ulong width, ulong height, ulong *p)
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
      while ((x < width) && (y < height)) {
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

    ulong ComputeBresenhamLineNX(ulong x, ulong y, long dx, long dy,
                                 ulong phase, ulong width, ulong height,
                                 ulong *p)
    /* Computes pixel coords along line with slope -1<m<0 */
    /* Returns # of pixel coords (num) written to array p (num <= width) */
    {
      ulong x0 = x, idx = y * width + x;
      long dp = 2 * dy - 2 * phase, twody = 2 * dy, twodydx = 2 * dy - 2 * dx;

      while (y >= height) {
        if (dp >= 0) {
          y--;
          dp += twodydx;
        } else
          dp += twody;
        x++;
      }
      if (x >= width)
        return (0);
      x0  = x;
      idx = y * width + x;
      *p  = idx;
      p++;
      while ((x < width - 1) && (y > 0)) {
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

    ulong ComputeBresenhamLinePY(long x, long y, long dx, long dy, ulong phase,
                                 ulong width, ulong height, ulong *p)
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
      while ((y < height) && (x < width)) {
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

    ulong ComputeBresenhamLineNY(ulong x, ulong y, long dx, long dy,
                                 ulong phase, ulong width, ulong height,
                                 ulong *p)
    /* Computes pixel coords along line with slope m<-1 */
    /* Returns # of pixel coords (num) written to array p (num <= height) */
    {
      ulong y0, idx;
      long dp = 2 * dx - 2 * phase, twodx = 2 * dx, twodxdy = 2 * dx - 2 * dy;

      while (x >= width) {
        if (dp >= 0) {
          x--;
          dp += twodxdy;
        } else
          dp += twodx;
        y++;
      }
      if (y >= height)
        return (0);
      y0  = y;
      idx = y * width + x;
      *p  = idx;
      p++;
      while ((y < height - 1) && (x > 0) && (x < width)) {
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

    template <class T1>
    void DilateHorLine(T1 *f, ulong width, ulong k, T1 *g, T1 *h, T1 *h2, T1 *r)
    /* k is length of SE in number of pixels */
    /* width is length of g, h, h2, and r */
    {
      ulong x, x2;

      for (x = 0; x < width; x++) {
        x2 = width - 1 - x;
        if (x % k)
          g[x] = MAX(g[x - 1], f[x]);
        else
          g[x] = f[x];

        if (((x2 % k) == (k - 1)) || (x2 == (width - 1)))
          h[x2] = f[x2];
        else
          h[x2] = MAX(h[x2 + 1], f[x2]);
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

    template <class T1>
    void DilateVerLine(T1 *f, ulong width, ulong height, ulong k, T1 *g, T1 *h,
                       T1 *h2, T1 *r)
    /* k is length of SE in number of pixels */
    /* height is length of g, h, h2, and r */
    {
      ulong y, y2;

      for (y = 0; y < height; y++) {
        y2 = height - 1 - y;
        if (y % k)
          g[y] = MAX(g[y - 1], f[y * width]);
        else
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

    template <class T1>
    void DilateLine(T1 *f, ulong width, ulong height, ulong k, ulong nx,
                    ulong *p, T1 *g, T1 *h, T1 *h2, T1 *r)
    /* k is length of SE in number of pixels */
    /* nx is length of p, g, h, and r */
    {
      ulong x, x2;

      for (x = 0; x < nx; x++) {
        x2 = nx - 1 - x;
        if (x % k)
          g[x] = MAX(g[x - 1], f[p[x]]);
        else
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

    template <class T1>
    void ImageGrayDilateHor(ImageGray *img, ulong k, T1 *g, T1 *h, T1 *h2,
                            ImageGray *out)
    {
      T1 *f       = (T1 *) img->Pixmap;
      T1 *r       = (T1 *) out->Pixmap;
      ulong width = img->Width, y;

      for (y = 0; y < img->Height; y++) {
        DilateHorLine(f, width, k, g, h, h2, r);
        f += width;
        r += width;
      }
    } /* ImageGrayDilateHor */

    template <class T1>
    void ImageGrayDilateVer(ImageGray *img, ulong k, T1 *g, T1 *h, T1 *h2,
                            ImageGray *out)
    {
      T1 *f       = (T1 *) img->Pixmap;
      T1 *r       = (T1 *) out->Pixmap;
      ulong width = img->Width, height = img->Height, x;

      for (x = 0; x < width; x++) {
        DilateVerLine(f, width, height, k, g, h, h2, r);
        f++;
        r++;
      }
    } /* ImageGrayDilateVer */

    template <class T1>
    void ImageGrayDilateLine(ImageGray *img, ulong k, long dx, long dy,
                             ulong phase, ulong *p, T1 *g, T1 *h, T1 *h2,
                             ImageGray *out)
    {
      T1 *f       = (T1 *) img->Pixmap;
      T1 *r       = (T1 *) out->Pixmap;
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

    template <class T1, class T2>
    RES_C t_ImFastLineDilate_Soille(const Image<T1> &imIn, const int angle,
                                    const int radius, Image<T2> &imOut)
    {
      // Check inputs
      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      int W, H;
      W = imIn.getWxSize();
      H = imIn.getWySize();

      int maxnx = MAX(W, H);
      T1 *g     = new T1[maxnx];
      T1 *h     = new T1[maxnx];
      T1 *h2    = new T1[maxnx];
      ulong *p  = new ulong[maxnx];

      ImageGray MyImgIn, MyImgOut;

      MyImgIn.Width  = W;
      MyImgIn.Height = H;
      MyImgIn.Pixmap = (void *) imIn.rawPointer();

      MyImgOut.Width  = W;
      MyImgOut.Height = H;
      MyImgOut.Pixmap = (void *) imOut.rawPointer();

      int dx = (int) (cos(angle * PI / 180.0) * maxnx);
      int dy = (int) (-sin(angle * PI / 180.0) * maxnx);

      if (dx != 0 && dy != 0) {
        // Be carreful with the boundaries of the picture:
        // With some angle, some pixel are not choosen... and their values are 0
        // So we copy the init picture
        T2 *bufferOut      = imOut.rawPointer();
        const T1 *bufferIn = imIn.rawPointer();
        for (int i = W * H - 1; i >= 0; i--)
          bufferOut[i] = (T2) bufferIn[i];
      }

      ImageGrayDilateLine(&MyImgIn, radius * 2 + 1, dx, dy, 1, p, g, h, h2,
                          &MyImgOut);

      delete[] g;
      delete[] h;
      delete[] h2;
      delete[] p;

      return RES_OK;
    }
  } // namespace FastLine
} // namespace morphee

#endif
