#ifndef __FAST_LINE_MORARD_T_HPP__
#define __FAST_LINE_MORARD_T_HPP__

// Vincent Morard
// 9 september 2010
// MAJ 8 ctober 2010  (in constant time wrt the radius)
// The algorithm used below is described in:
// Morard, Dokladal, Decenciere, "One-dimensional openings, granulometries and
// component trees in O (1) per pixel", IEEE Journal of Selected Topics in
// Signal Processing 6 (7), 840-848, 2011

#include "Core/include/DCore.h"

namespace smil
{
  template <typename T>
  class MorardLineMorpho
  {
  public:
    MorardLineMorpho() {};
    ~MorardLineMorpho() {};

  private:
    struct Node {
      // c'etait unsigned int
      int           StartPos;
      unsigned char Passed;
      T             Value;
    };

    int wp, size, *LineIdx; // Writing position

    inline void BuildOpeningFromPoint(T F, int Ind, Node *MyStack,
                                      int *stackSize, T *bufferOut)
    {
      LineIdx[wp] = Ind;

      // -1-  Si la pile est vide ou si on a un front montant, on empile
      // if(*stackSize == 0 || F > MyStack[*stackSize-1].Value){
      // WE ASSUME THAT WE HAVE PUSH THE FIRST PIXEL OF THE LINE!!
      if (F > MyStack[*stackSize - 1].Value) {
        (MyStack[*stackSize]).StartPos = wp;
        (MyStack[*stackSize]).Passed   = 0;
        (MyStack[*stackSize]).Value    = F;
        *stackSize += 1;
      } else {
        while (F < (MyStack[*stackSize - 1]).Value) {
          // On depile... then MyStack[*stackSize]-->NodeOut
          *stackSize -= 1;

          // We have passed the criteria
          if (MyStack[*stackSize].Passed ||
              (wp - (MyStack[*stackSize]).StartPos >= size)) {
            for (int j = 0; j < *stackSize; j++) {
              for (int i = (MyStack[j]).StartPos; i < (MyStack[j + 1]).StartPos;
                   i++)
                bufferOut[LineIdx[i]] = (MyStack[j]).Value;
            }
            for (int i = (MyStack[*stackSize]).StartPos; i < wp; i++)
              bufferOut[LineIdx[i]] = (MyStack[*stackSize]).Value;

            (MyStack[0]).StartPos = wp;
            (MyStack[0]).Passed   = 1;
            (MyStack[0]).Value    = F;
            *stackSize            = 1;
            break;
          }

          if (*stackSize == 0 || F > (MyStack[*stackSize - 1]).Value) {
            (MyStack[*stackSize]).Value = F;
            *stackSize += 1;
            break;
          }
        }
      }
      wp++;
    }

    void EndProcess(Node *MyStack, int *stackSize, T *bufferOut)
    {
      while (*stackSize != 0) {
        if (wp - (MyStack[*stackSize - 1]).StartPos >= size) {
          for (int j = 0; j < *stackSize - 1; j++) {
            for (int i = (MyStack[j]).StartPos; i < (MyStack[j + 1]).StartPos;
                 i++)
              bufferOut[LineIdx[i]] = (MyStack[j]).Value;
          }

          for (int i = (MyStack[*stackSize - 1]).StartPos; i < wp; i++)
            bufferOut[LineIdx[i]] = (MyStack[*stackSize - 1]).Value;
          *stackSize = 0;
          return;
        }
        *stackSize -= 1;
      }

      // La ligne ne respecte pas le critère:
      for (int i = (MyStack[0]).StartPos; i < wp; i++)
        bufferOut[LineIdx[i]] = (MyStack[0]).Value;
    }

    int ComputeLinePosDiag(T *bufferIn, int W, int H, int x, int y,
                           Node *MyStack, int *stackSize, T *bufferOut)
    {
      int idx;
      int x0;

      if (x < 0) {
        y -= x;
        x = 0;
      }
      if (y < 0) {
        x -= y;
        y = 0;
      }
      x0  = x;
      idx = y * W + x;

      if ((x < W) && (y < H)) {
        wp                  = 1;
        MyStack[0].Passed   = 0;
        MyStack[0].Value    = bufferIn[idx];
        MyStack[0].StartPos = 0;
        *stackSize          = 1;
        LineIdx[0]          = idx;

        // Next pixel
        idx += W + 1;
        y++;
      }

      for (; (x < W) && (y < H); x++) {
        BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize,
                              bufferOut);
        idx += W + 1;
        y++;
      }
      return (x - x0);
    }

    int ComputeLineNegDiag(T *bufferIn, int W, int H, int x, int y,
                           Node *MyStack, int *stackSize, T *bufferOut)
    {
      int idx;
      int x0;

      if (y >= H) {
        x += y - H + 1;
        y = H - 1;
      }
      if (x >= W)
        return (0);
      x0  = x;
      idx = y * W + x;

      wp                  = 1;
      MyStack[0].Passed   = 0;
      MyStack[0].Value    = bufferIn[idx];
      MyStack[0].StartPos = 0;
      *stackSize          = 1;
      LineIdx[0]          = idx;

      // The first one is already push
      // BuildOpeningFromPoint(bufferIn[idx],idx,MyStack,stackSize,bufferOut);

      while ((x < W - 1) && (y > 0)) {
        // p++;
        x++;
        y--;
        idx -= W - 1;

        BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize,
                              bufferOut);
      }
      return (x - x0 + 1);
    }

    int ComputeBresenhamLinePX(T *bufferIn, int W, int H, int x, int y, int dx,
                               int dy, Node *MyStack, int *stackSize,
                               T *bufferOut)
    {
      int idx;
      int x0;
      int dp = 2 * dy - 2, twody = 2 * dy, twodydx = 2 * dy - 2 * dx;

      while ((x < 0) || (y < 0)) {
        if (dp >= 0) {
          y++;
          dp += twodydx;
        } else
          dp += twody;
        x++;
      }
      x0  = x;
      idx = y * W + x;

      if (x >= 0 && x < W && y >= 0 && y < H) {
        wp                  = 1;
        MyStack[0].Passed   = 0;
        MyStack[0].Value    = bufferIn[idx];
        MyStack[0].StartPos = 0;
        *stackSize          = 1;
        LineIdx[0]          = idx;

        // Next pixel, The first one is already push
        if (dp >= 0) {
          y++;
          idx += W;
          dp += twodydx;
        } else
          dp += twody;
        x++;
        idx++;
      }

      while ((x < W) && (y < H)) {
        BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize,
                              bufferOut);

        if (dp >= 0) {
          y++;
          idx += W;
          dp += twodydx;
        } else
          dp += twody;
        x++;
        idx++;
      }
      return (x - x0);
    }

    int ComputeBresenhamLineNX(T *bufferIn, int W, int H, int x, int y, int dx,
                               int dy, Node *MyStack, int *stackSize,
                               T *bufferOut)
    {
      int x0 = x, idx = y * W + x;
      int dp = 2 * dy - 2, twody = 2 * dy, twodydx = 2 * dy - 2 * dx;

      while (y >= H) {
        if (dp >= 0) {
          y--;
          dp += twodydx;
        } else
          dp += twody;
        x++;
      }
      if (x >= W)
        return (0);
      x0  = x;
      idx = y * W + x;

      wp                  = 1;
      MyStack[0].Passed   = 0;
      MyStack[0].Value    = bufferIn[idx];
      MyStack[0].StartPos = 0;
      *stackSize          = 1;
      LineIdx[0]          = idx;
      // The first one is already push
      // BuildOpeningFromPoint(bufferIn[idx],idx,MyStack,stackSize,bufferOut);

      while ((x < W - 1) && (y > 0)) {
        if (dp >= 0) {
          y--;
          idx -= W;
          dp += twodydx;
        } else
          dp += twody;
        x++;
        idx++;

        BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize,
                              bufferOut);
      }
      return (x - x0 + 1);
    }

    int ComputeBresenhamLinePY(T *bufferIn, int W, int H, int x, int y, int dx,
                               int dy, Node *MyStack, int *stackSize,
                               T *bufferOut)
    {
      int y0, idx;
      int dp = 2 * dx - 2, twodx = 2 * dx, twodxdy = 2 * dx - 2 * dy;

      while ((x < 0) || (y < 0)) {
        if (dp >= 0) {
          x++;
          dp += twodxdy;
        } else
          dp += twodx;
        y++;
      }
      y0  = y;
      idx = y * W + x;

      // We push the first pixel
      if (x >= 0 && x < W && y >= 0 && y < H) {
        wp                  = 1;
        MyStack[0].Passed   = 0;
        MyStack[0].Value    = bufferIn[idx];
        MyStack[0].StartPos = 0;
        *stackSize          = 1;
        LineIdx[0]          = idx;

        // Next pixel, the first one is already pushed
        if (dp >= 0) {
          x++;
          idx++;
          dp += twodxdy;
        } else
          dp += twodx;
        y++;
        idx += W;
      }

      while ((y < H) && (x < W)) {
        BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize,
                              bufferOut);
        if (dp >= 0) {
          x++;
          idx++;
          dp += twodxdy;
        } else
          dp += twodx;
        y++;
        idx += W;
      }
      return (y - y0);
    }

    int ComputeBresenhamLineNY(T *bufferIn, int W, int H, int x, int y, int dx,
                               int dy, Node *MyStack, int *stackSize,
                               T *bufferOut)
    {
      int y0, idx;
      int dp = 2 * dx - 2, twodx = 2 * dx, twodxdy = 2 * dx - 2 * dy;

      while (x >= W) {
        if (dp >= 0) {
          x--;
          dp += twodxdy;
        } else
          dp += twodx;
        y++;
      }
      if (y >= H)
        return (0);
      y0  = y;
      idx = y * W + x;

      wp                  = 1;
      MyStack[0].Passed   = 0;
      MyStack[0].Value    = bufferIn[idx];
      MyStack[0].StartPos = 0;
      *stackSize          = 1;
      LineIdx[0]          = idx;
      // Next pixel, the first one is already pushed
      // BuildOpeningFromPoint(bufferIn[idx],idx,MyStack,stackSize,bufferOut);

      while ((y < H - 1) && (x > 0) && (x < W)) {
        if (dp >= 0) {
          x--;
          idx--;
          dp += twodxdy;
        } else
          dp += twodx;
        y++;
        idx += W;

        BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize,
                              bufferOut);
      }
      return (y - y0 + 1);
    }

    void LineOpeningHorz(T *bufferIn, int W, int H, int radius, T *bufferOut)
    {
      size = radius * 2 + 1;
      // Pas génant de faire <T2> puisque T1 est du même type que T2 mais T1 est
      // const...

      Node MyStack[256];
      int  stackPos = 0;

      int i, j;

      for (j = 0; j < H; j++) // Pour toutes les lignes
      {
        wp                  = 1;
        MyStack[0].Passed   = 0;
        MyStack[0].Value    = bufferIn[j * W];
        MyStack[0].StartPos = 0;
        stackPos            = 1;
        LineIdx[0]          = j * W;
        for (i = 1; i < W; i++) // Pour toutes les colonnes
          BuildOpeningFromPoint(bufferIn[i + j * W], i + j * W, MyStack,
                                &stackPos, bufferOut);
        EndProcess(MyStack, &stackPos, bufferOut);
      }
    }

    void LineOpeningVert(T *bufferIn, int W, int H, int radius, T *bufferOut)
    {
      size = radius * 2 + 1;

      Node MyStack[256];
      int  stackPos = 0;

      int i, j;

      for (i = 0; i < W; i++) // Pour toutes les colonnes
      {
        wp                  = 1;
        MyStack[0].Passed   = 0;
        MyStack[0].Value    = bufferIn[i];
        MyStack[0].StartPos = 0;
        stackPos            = 1;
        LineIdx[0]          = i;
        for (j = 1; j < H; j++) // Pour toutes les lignes
          BuildOpeningFromPoint(bufferIn[i + j * W], i + j * W, MyStack,
                                &stackPos, bufferOut);
        EndProcess(MyStack, &stackPos, bufferOut);
      }
    }

    void LineOpeningDiag(T *bufferIn, int W, int H, int dx, int dy, int radius,
                         T *bufferOut)
    {
      size = radius * 2 + 1;
      int y, x, nx;

      Node MyStack[256];
      int  stackPos = 0;

      if (abs(dx) == abs(dy)) {
        if (dx == -dy) {
          y = 0;
          do {
            nx = ComputeLineNegDiag(bufferIn, W, H, 0, y++, MyStack, &stackPos,
                                    bufferOut);
            EndProcess(MyStack, &stackPos, bufferOut);
          } while (nx > 0);

        } else {
          y = H - 2; //-2 to avoid a bug (empty image)
          do {
            nx = ComputeLinePosDiag(bufferIn, W, H, 0, y--, MyStack, &stackPos,
                                    bufferOut);
            EndProcess(MyStack, &stackPos, bufferOut);
          } while (nx > 0);
        }
      } else if (abs(dx) > abs(dy)) {
        if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
          dx = abs(dx);
          dy = abs(dy);

          y = H - 1;
          do {
            nx = ComputeBresenhamLinePX(bufferIn, W, H, 0, y--, dx, dy, MyStack,
                                        &stackPos, bufferOut);
            EndProcess(MyStack, &stackPos, bufferOut);
          } while (nx > 0);
        } else {
          dx = abs(dx);
          dy = abs(dy);

          y = 0;
          do {
            nx = ComputeBresenhamLineNX(bufferIn, W, H, 0, y++, dx, dy, MyStack,
                                        &stackPos, bufferOut);
            EndProcess(MyStack, &stackPos, bufferOut);
          } while (nx > 0);
        }
      } else {
        if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
          dx = abs(dx);
          dy = abs(dy);

          x = W - 1;
          do {
            nx = ComputeBresenhamLinePY(bufferIn, W, H, x--, 0, dx, dy, MyStack,
                                        &stackPos, bufferOut);
            EndProcess(MyStack, &stackPos, bufferOut);
          } while (nx > 0);
        } else {
          dx = abs(dx);
          dy = abs(dy);

          x = 0;
          do {
            nx = ComputeBresenhamLineNY(bufferIn, W, H, x++, 0, dx, dy, MyStack,
                                        &stackPos, bufferOut);
            EndProcess(MyStack, &stackPos, bufferOut);
          } while (nx > 0);
        }
      }
    }

    ;

  public:
    RES_T ImFastLineOpen(const Image<T> &imIn, const int angle,
                         const int radius, Image<T> &imOut)
    {
      // Check inputs
      ASSERT_ALLOCATED(&imIn);
      ASSERT_ALLOCATED(&imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      int W, H;

      W = imIn.getWidth();
      H = imIn.getHeight();

      typename Image<T>::lineType bufferIn  = imIn.getPixels();
      typename Image<T>::lineType bufferOut = imOut.getPixels();

      int rd = (int) (angle * PI / 180.);
      int r  = (int) (radius * std::max(fabs(cos(rd)), fabs(sin(rd))) + 0.5);

      int maxnx = std::max(W, H);
      LineIdx   = new int[maxnx + 3];
      int dx    = (int) (cos(angle * PI / 180.0) * maxnx);
      int dy    = (int) (-sin(angle * PI / 180.0) * maxnx);

      if (dx == 0)
        LineOpeningVert(bufferIn, W, H, r, bufferOut);
      else if (dy == 0)
        LineOpeningHorz(bufferIn, W, H, r, bufferOut);
      else
        LineOpeningDiag(bufferIn, W, H, dx, dy, r, bufferOut);

      delete[] LineIdx;
      return RES_OK;
    }
  };

  //
  //
  //  Start of global code
  //
  template <class T>
  RES_T imFastLineOpen(const Image<T> &imIn, const int angle, const int radius,
                       Image<T> &imOut)
  {
    MorardLineMorpho<T> morard;

    return morard.ImFastLineOpen(imIn, angle, radius, imOut);
  }

} // namespace smil

#endif
