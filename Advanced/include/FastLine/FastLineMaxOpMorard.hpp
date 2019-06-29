#ifndef __FAST_LINE_MORARD_MAXOP_T_HPP__
#define __FAST_LINE_MORARD_MAXOP_T_HPP__

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/morpheeImage.hpp>

// Vincent Morard
// 11 octobre : max opening implementation (spetialization to avoid
// aritSupImage())

namespace morphee
{
  namespace FastLine
  {
    int Orientation;

    template <class T1, class T1bis, class T2>
    inline void BuildOpeningFromPoint_maxOp(T1 F, int Ind, Node<T1bis> *MyStack,
                                            int *stackSize, T2 *bufferOut,
                                            T2 *bufferOri = 0)
    {
      LineIdx[wp] = Ind;

      // -1-  Si la pile est vide ou si on a un front montant, on empile
      if (*stackSize == 0 || F > MyStack[*stackSize - 1].Value) {
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
                   i++) {
                if (bufferOri != 0 && (MyStack[j]).Value != 0 &&
                    (MyStack[j]).Value > bufferOut[LineIdx[i]])
                  bufferOri[LineIdx[i]] = Orientation;
                bufferOut[LineIdx[i]] =
                    MAX((MyStack[j]).Value, bufferOut[LineIdx[i]]);
              }
            }
            for (int i = (MyStack[*stackSize]).StartPos; i < wp; i++) {
              if (bufferOri != 0 && (MyStack[*stackSize]).Value != 0 &&
                  (MyStack[*stackSize]).Value > bufferOut[LineIdx[i]])
                bufferOri[LineIdx[i]] = Orientation;
              bufferOut[LineIdx[i]] =
                  MAX((MyStack[*stackSize]).Value, bufferOut[LineIdx[i]]);
            }
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

    template <typename T1bis, class T2>
    void EndProcess_maxOp(Node<T1bis> *MyStack, int *stackSize, T2 *bufferOut,
                          T2 *bufferOri)
    {
      while (*stackSize != 0) {
        if (wp - (MyStack[*stackSize - 1]).StartPos >= size) {
          for (int j = 0; j < *stackSize - 1; j++) {
            for (int i = (MyStack[j]).StartPos; i < (MyStack[j + 1]).StartPos;
                 i++) {
              if (bufferOri != 0 && (MyStack[j]).Value != 0 &&
                  (MyStack[j]).Value > bufferOut[LineIdx[i]])
                bufferOri[LineIdx[i]] = Orientation;
              bufferOut[LineIdx[i]] =
                  MAX((MyStack[j]).Value, bufferOut[LineIdx[i]]);
            }
          }

          for (int i = (MyStack[*stackSize - 1]).StartPos; i < wp; i++) {
            if (bufferOri != 0 && (MyStack[*stackSize - 1]).Value != 0 &&
                (MyStack[*stackSize - 1]).Value > bufferOut[LineIdx[i]])
              bufferOri[LineIdx[i]] = Orientation;
            bufferOut[LineIdx[i]] =
                MAX((MyStack[*stackSize - 1]).Value, bufferOut[LineIdx[i]]);
          }

          *stackSize = 0;
          return;
        }
        *stackSize -= 1;
      }

      // La ligne ne respecte pas le critère:
      for (int i = (MyStack[0]).StartPos; i < wp; i++) {
        if (bufferOri != 0 && (MyStack[0]).Value != 0 &&
            (MyStack[0]).Value > bufferOut[LineIdx[i]])
          bufferOri[LineIdx[i]] = Orientation;
        bufferOut[LineIdx[i]] = MAX((MyStack[0]).Value, bufferOut[LineIdx[i]]);
      }
    }

    template <class T1, typename T1bis, class T2>
    int ComputeLinePosDiag_maxOp(T1 *bufferIn, int W, int H, int x, int y,
                                 Node<T1bis> *MyStack, int *stackSize,
                                 T2 *bufferOut, T2 *bufferOri = 0)
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
      for (; (x < W) && (y < H); x++) {
        BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                    bufferOut, bufferOri);
        idx += W + 1;
        // p++;
        y++;
      }
      return (x - x0);
    }

    template <class T1, typename T1bis, class T2>
    int ComputeLineNegDiag_maxOp(T1 *bufferIn, int W, int H, int x, int y,
                                 Node<T1bis> *MyStack, int *stackSize,
                                 T2 *bufferOut, T2 *bufferOri = 0)
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

      BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                  bufferOut, bufferOri);
      while ((x < W - 1) && (y > 0)) {
        // p++;
        x++;
        y--;
        idx -= W - 1;

        BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                    bufferOut, bufferOri);
      }
      return (x - x0 + 1);
    }

    template <class T1, typename T1bis, class T2>
    int ComputeBresenhamLinePX_maxOp(T1 *bufferIn, int W, int H, int x, int y,
                                     int dx, int dy, Node<T1bis> *MyStack,
                                     int *stackSize, T2 *bufferOut,
                                     T2 *bufferOri = 0)
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
      while ((x < W) && (y < H)) {
        BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                    bufferOut, bufferOri);

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

    template <class T1, typename T1bis, class T2>
    int ComputeBresenhamLineNX_maxOp(T1 *bufferIn, int W, int H, int x, int y,
                                     int dx, int dy, Node<T1bis> *MyStack,
                                     int *stackSize, T2 *bufferOut,
                                     T2 *bufferOri = 0)
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

      BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                  bufferOut, bufferOri);

      while ((x < W - 1) && (y > 0)) {
        if (dp >= 0) {
          y--;
          idx -= W;
          dp += twodydx;
        } else
          dp += twody;
        x++;
        idx++;

        BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                    bufferOut, bufferOri);
      }
      return (x - x0 + 1);
    }

    template <class T1, typename T1bis, class T2>
    int ComputeBresenhamLinePY_maxOp(T1 *bufferIn, int W, int H, int x, int y,
                                     int dx, int dy, Node<T1bis> *MyStack,
                                     int *stackSize, T2 *bufferOut,
                                     T2 *bufferOri = 0)
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
      while ((y < H) && (x < W)) {
        BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                    bufferOut, bufferOri);
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

    template <class T1, typename T1bis, class T2>
    int ComputeBresenhamLineNY_maxOp(T1 *bufferIn, int W, int H, int x, int y,
                                     int dx, int dy, Node<T1bis> *MyStack,
                                     int *stackSize, T2 *bufferOut,
                                     T2 *bufferOri = 0)
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

      BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                  bufferOut, bufferOri);

      while ((y < H - 1) && (x > 0) && (x < W)) {
        if (dp >= 0) {
          x--;
          idx--;
          dp += twodxdy;
        } else
          dp += twodx;
        y++;
        idx += W;

        BuildOpeningFromPoint_maxOp(bufferIn[idx], idx, MyStack, stackSize,
                                    bufferOut, bufferOri);
      }
      return (y - y0 + 1);
    }

    template <class T1, class T2>
    void LineOpeningH_maxOp(T1 *bufferIn, int W, int H, int radius,
                            T2 *bufferOut, T2 *bufferOri = 0)
    {
      size = radius * 2 + 1;
      // Pas génant de faire <T2> puisque T1 est du même type que T2 mais T1 est
      // const...

      Node<T2> MyStack[256];
      int stackPos = 0;
      int i, j;

      for (j = 0; j < H; j++) // Pour toutes les lignes
      {
        wp = 0;
        for (i = 0; i < W; i++) // Pour toutes les colonnes
          BuildOpeningFromPoint_maxOp(bufferIn[i + j * W], i + j * W, MyStack,
                                      &stackPos, bufferOut, bufferOri);
        EndProcess_maxOp(MyStack, &stackPos, bufferOut, bufferOri);
      }
    }

    template <class T1, class T2>
    void LineOpeningV_maxOp(T1 *bufferIn, int W, int H, int radius,
                            T2 *bufferOut, T2 *bufferOri = 0)
    {
      size = radius * 2 + 1;

      Node<T2> MyStack[256];
      int stackPos = 0;

      int i, j;

      for (i = 0; i < W; i++) // Pour toutes les colonnes
      {
        wp = 0;
        for (j = 0; j < H; j++) // Pour toutes les lignes
          BuildOpeningFromPoint_maxOp(bufferIn[i + j * W], i + j * W, MyStack,
                                      &stackPos, bufferOut, bufferOri);
        EndProcess_maxOp(MyStack, &stackPos, bufferOut, bufferOri);
      }
    }

    template <class T1, class T2>
    void LineOpeningDiag_maxOp(T1 *bufferIn, int W, int H, int dx, int dy,
                               int radius, T2 *bufferOut, T2 *bufferOri = 0)
    {
      size = radius * 2 + 1;
      int y, x, nx;

      Node<T2> MyStack[256];
      int stackPos = 0;

      if (abs(dx) == abs(dy)) {
        if (dx == -dy) {
          y = 0;
          do {
            wp = 0;
            nx = ComputeLineNegDiag_maxOp(bufferIn, W, H, 0, y++, MyStack,
                                          &stackPos, bufferOut, bufferOri);
            EndProcess_maxOp(MyStack, &stackPos, bufferOut, bufferOri);
          } while (nx > 0);

        } else {
          y = H - 2;
          do {
            wp = 0;
            nx = ComputeLinePosDiag_maxOp(bufferIn, W, H, 0, y--, MyStack,
                                          &stackPos, bufferOut, bufferOri);
            EndProcess_maxOp(MyStack, &stackPos, bufferOut, bufferOri);
          } while (nx > 0);
        }
      } else if (abs(dx) > abs(dy)) {
        if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
          dx = abs(dx);
          dy = abs(dy);

          y = H - 1;
          do {
            wp = 0;
            nx = ComputeBresenhamLinePX_maxOp(bufferIn, W, H, 0, y--, dx, dy,
                                              MyStack, &stackPos, bufferOut,
                                              bufferOri);
            EndProcess_maxOp(MyStack, &stackPos, bufferOut, bufferOri);
          } while (nx > 0);
        } else {
          dx = abs(dx);
          dy = abs(dy);

          y = 0;
          do {
            wp = 0;
            nx = ComputeBresenhamLineNX_maxOp(bufferIn, W, H, 0, y++, dx, dy,
                                              MyStack, &stackPos, bufferOut,
                                              bufferOri);
            EndProcess_maxOp(MyStack, &stackPos, bufferOut, bufferOri);
          } while (nx > 0);
        }
      } else {
        if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
          dx = abs(dx);
          dy = abs(dy);

          x = W - 1;
          do {
            wp = 0;
            nx = ComputeBresenhamLinePY_maxOp(bufferIn, W, H, x--, 0, dx, dy,
                                              MyStack, &stackPos, bufferOut,
                                              bufferOri);
            EndProcess_maxOp(MyStack, &stackPos, bufferOut, bufferOri);
          } while (nx > 0);
        } else {
          dx = abs(dx);
          dy = abs(dy);

          x = 0;
          do {
            wp = 0;
            nx = ComputeBresenhamLineNY_maxOp(bufferIn, W, H, x++, 0, dx, dy,
                                              MyStack, &stackPos, bufferOut,
                                              bufferOri);
            EndProcess_maxOp(MyStack, &stackPos, bufferOut, bufferOri);
          } while (nx > 0);
        }
      }
    }

    template <class T1, class T2>
    RES_C t_ImFastLineMaxOpen_Morard(const Image<T1> &imIn, const int nbAngle,
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
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();
      memset(bufferOut, 0, W * H * sizeof(T2));
      int maxnx = MAX(W, H);
      LineIdx   = new int[maxnx + 3];

      int angle, dx, dy;
      for (int k = 0; k < nbAngle; k++) {
        angle = (int) (k * 180 / (double) nbAngle);
        dx    = (int) (cos(angle * PI / 180.0) * maxnx);
        dy    = (int) (-sin(angle * PI / 180.0) * maxnx);

        if (dx == 0)
          LineOpeningV_maxOp(bufferIn, W, H, radius, bufferOut);
        else if (dy == 0)
          LineOpeningH_maxOp(bufferIn, W, H, radius, bufferOut);
        else
          LineOpeningDiag_maxOp(bufferIn, W, H, dx, dy, radius, bufferOut);
      }

      delete[] LineIdx;
      return RES_OK;
    }

    template <class T1, class T2>
    RES_C t_ImFastLineMaxOpenOrientation_Morard(const Image<T1> &imIn,
                                                const int nbAngle,
                                                const int radius,
                                                Image<T2> &imOut)
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
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();

      Image<T1> imTmp;
      imTmp.setSize(W, H);
      imTmp.allocateImage();
      T1 *bufferTmp = imTmp.rawPointer();
      memset(bufferOut, 0, W * H * sizeof(T1));
      memset(bufferTmp, 0, W * H * sizeof(T2));

      int maxnx = MAX(W, H);
      LineIdx   = new int[maxnx + 3];

      // First one : the horizontal opening to have a reference for the MAX
      Orientation = 1;
      LineOpeningH_maxOp(bufferIn, W, H, radius, bufferTmp, bufferOut);

      int angle, dx, dy;
      for (int k = 1; k < nbAngle; k++) {
        angle       = (int) (k * 180 / (double) nbAngle);
        dx          = (int) (cos(angle * PI / 180.0) * maxnx);
        dy          = (int) (-sin(angle * PI / 180.0) * maxnx);
        Orientation = angle + 1;
        if (dx == 0)
          LineOpeningV_maxOp(bufferIn, W, H, radius, bufferTmp, bufferOut);
        else if (dy == 0)
          LineOpeningH_maxOp(bufferIn, W, H, radius, bufferTmp, bufferOut);
        else
          LineOpeningDiag_maxOp(bufferIn, W, H, dx, dy, radius, bufferTmp,
                                bufferOut);
      }

      delete[] LineIdx;
      imTmp.deallocateImage();
      return RES_OK;
    }

  } // namespace FastLine
} // namespace morphee

#endif
