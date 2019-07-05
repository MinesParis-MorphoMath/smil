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
  template <typename T> struct Node {
      // c'etait unsigned int
     int StartPos;
    unsigned char Passed;
    T Value;
  };

  int wp, size, *LineIdx; // Writing position

  template <class T>
  inline void BuildOpeningFromPoint(T F, int Ind, Node<T> *MyStack,
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

  template <typename T>
  void EndProcess(Node<T> *MyStack, int *stackSize, T *bufferOut)
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

    // La ligne ne respecte pas le crit�re:
    for (int i = (MyStack[0]).StartPos; i < wp; i++)
      bufferOut[LineIdx[i]] = (MyStack[0]).Value;
  }

  template <class T>
  int ComputeLinePosDiag_v2(T *bufferIn, int W, int H, int x, int y,
                            Node<T> *MyStack, int *stackSize, T *bufferOut)
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
      BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize, bufferOut);
      idx += W + 1;
      y++;
    }
    return (x - x0);
  }

  template <class T1, typename T1bis, class T2>
  int ComputeLineNegDiag_v2(T1 *bufferIn, int W, int H, int x, int y,
                            Node<T1bis> *MyStack, int *stackSize, T2 *bufferOut)
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

      BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize, bufferOut);
    }
    return (x - x0 + 1);
  }

  template <class T>
  int ComputeBresenhamLinePX_v2(T *bufferIn, int W, int H, int x, int y, int dx,
                                int dy, Node<T> *MyStack, int *stackSize,
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
      BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize, bufferOut);

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

  template <class T>
  int ComputeBresenhamLineNX_v2(T *bufferIn, int W, int H, int x, int y, int dx,
                                int dy, Node<T> *MyStack, int *stackSize,
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

      BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize, bufferOut);
    }
    return (x - x0 + 1);
  }

  template <class T>
  int ComputeBresenhamLinePY_v2(T *bufferIn, int W, int H, int x, int y, int dx,
                                int dy, Node<T> *MyStack, int *stackSize,
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
      BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize, bufferOut);
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

  template <class T>
  int ComputeBresenhamLineNY_v2(T *bufferIn, int W, int H, int x, int y, int dx,
                                int dy, Node<T> *MyStack, int *stackSize,
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

      BuildOpeningFromPoint(bufferIn[idx], idx, MyStack, stackSize, bufferOut);
    }
    return (y - y0 + 1);
  }

  template <class T>
  void LineOpeningH_v2(T *bufferIn, int W, int H, int radius, T *bufferOut)
  {
    size = radius * 2 + 1;
    // Pas g�nant de faire <T2> puisque T1 est du m�me type que T2 mais T1 est
    // const...

    Node<T> MyStack[256];
    int stackPos = 0;

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

  template <class T>
  void LineOpeningV_v2(T *bufferIn, int W, int H, int radius, T *bufferOut)
  {
    size = radius * 2 + 1;

    Node<T> MyStack[256];
    int stackPos = 0;

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

  template <class T>
  void LineOpeningDiag_v2(T *bufferIn, int W, int H, int dx, int dy, int radius,
                          T *bufferOut)
  {
    size = radius * 2 + 1;
    int y, x, nx;

    Node<T> MyStack[256];
    int stackPos = 0;

    if (abs(dx) == abs(dy)) {
      if (dx == -dy) {
        y = 0;
        do {
          nx = ComputeLineNegDiag_v2(bufferIn, W, H, 0, y++, MyStack, &stackPos,
                                     bufferOut);
          EndProcess(MyStack, &stackPos, bufferOut);
        } while (nx > 0);

      } else {
        y = H - 2; //-2 to avoid a bug (empty image)
        do {
          nx = ComputeLinePosDiag_v2(bufferIn, W, H, 0, y--, MyStack, &stackPos,
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
          nx = ComputeBresenhamLinePX_v2(bufferIn, W, H, 0, y--, dx, dy,
                                         MyStack, &stackPos, bufferOut);
          EndProcess(MyStack, &stackPos, bufferOut);
        } while (nx > 0);
      } else {
        dx = abs(dx);
        dy = abs(dy);

        y = 0;
        do {
          nx = ComputeBresenhamLineNX_v2(bufferIn, W, H, 0, y++, dx, dy,
                                         MyStack, &stackPos, bufferOut);
          EndProcess(MyStack, &stackPos, bufferOut);
        } while (nx > 0);
      }
    } else {
      if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
        dx = abs(dx);
        dy = abs(dy);

        x = W - 1;
        do {
          nx = ComputeBresenhamLinePY_v2(bufferIn, W, H, x--, 0, dx, dy,
                                         MyStack, &stackPos, bufferOut);
          EndProcess(MyStack, &stackPos, bufferOut);
        } while (nx > 0);
      } else {
        dx = abs(dx);
        dy = abs(dy);

        x = 0;
        do {
          nx = ComputeBresenhamLineNY_v2(bufferIn, W, H, x++, 0, dx, dy,
                                         MyStack, &stackPos, bufferOut);
          EndProcess(MyStack, &stackPos, bufferOut);
        } while (nx > 0);
      }
    }
  }

  // Specialisation of the algo for horizontal images
  template <class T>
  RES_T ImFastLineOpen_Morard(const Image<T> &imIn, const int angle,
                              const int radius, Image<T> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn);
    ASSERT_ALLOCATED(&imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    int W, H;

    W                 = imIn.getWidth();
    H                 = imIn.getHeight();

    typename Image<T>::lineType bufferIn = imIn.getPixels();
    typename Image<T>::lineType bufferOut = imIn.getPixels();

    int maxnx = MAX(W, H);
    LineIdx   = new int[maxnx + 3];
    int dx    = (int) (cos(angle * PI / 180.0) * maxnx);
    int dy    = (int) (-sin(angle * PI / 180.0) * maxnx);

    if (dx == 0)
      LineOpeningV_v2(bufferIn, W, H, radius, bufferOut);
    else if (dy == 0)
      LineOpeningH_v2(bufferIn, W, H, radius, bufferOut);
    else
      LineOpeningDiag_v2(bufferIn, W, H, dx, dy, radius, bufferOut);

    delete[] LineIdx;
    return RES_OK;
  }

#if 0
    // Specialisation of the algo for horizontal images
    template <class T>
    RES_T ImFastLineOpeningH_v2(const Image<T> &imIn, const int radius,
                                  Image<T> &imOut)
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

      int W, H, j, rp, stackSize, size = radius * 2 + 1;
      Node<UINT8> MyStack[256];

      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();

      const T1 *F = &(bufferIn[0]);

      for (j = 0; j < H; j++) {
        stackSize           = 1;
        MyStack[0].Value    = *F;
        MyStack[0].StartPos = 0;
        MyStack[0].Passed   = 0;
        F += 1;
        for (rp = 1; rp < W; rp++, F += 1) {
          if (*F > MyStack[stackSize - 1].Value) {
            (MyStack[stackSize]).StartPos = rp;
            (MyStack[stackSize]).Passed   = 0;
            (MyStack[stackSize++]).Value  = *F;

          } else {
            while (*F < (MyStack[stackSize - 1]).Value) {
              // On depile... then MyStack[*stackSize]-->NodeOut
              stackSize--;

              // We have passed the criteria
              if (MyStack[stackSize].Passed ||
                  (rp - (MyStack[stackSize]).StartPos >= size)) {
                for (int k = 0; k < stackSize; k++) {
                  memset(&bufferOut[(MyStack[k]).StartPos + j * W],
                         (MyStack[k]).Value,
                         (MyStack[k + 1]).StartPos - (MyStack[k]).StartPos);
                  // for(int
                  // i=(MyStack[k]).StartPos;i<(MyStack[k+1]).StartPos;i++)
                  //	bufferOut[i+j*W] = (MyStack[k]).Value;
                }
                memset(&bufferOut[(MyStack[stackSize]).StartPos + j * W],
                       (MyStack[stackSize]).Value,
                       rp - (MyStack[stackSize]).StartPos);
                // for(int i=(MyStack[stackSize]).StartPos;i<rp;i++)
                //	bufferOut[i+j*W] = (MyStack[stackSize]).Value;

                (MyStack[0]).StartPos = rp;
                (MyStack[0]).Passed   = 1;
                (MyStack[0]).Value    = *F;
                stackSize             = 1;
                break;
              }

              if (stackSize == 0 || *F > (MyStack[stackSize - 1]).Value) {
                (MyStack[stackSize++]).Value = *F;
                break;
              }
            }
          }
        }
        // End of the line
        if (rp - (MyStack[0]).StartPos < size) {
          memset(&bufferOut[(MyStack[0]).StartPos + j * W], (MyStack[0]).Value,
                 rp - (MyStack[0]).StartPos);
        } else {
          while (stackSize != 0) {
            if (rp - (MyStack[stackSize - 1]).StartPos >= size) {
              for (int k = 0; k < stackSize - 1; k++) {
                memset(&bufferOut[(MyStack[k]).StartPos + j * W],
                       (MyStack[k]).Value,
                       (MyStack[k + 1]).StartPos - (MyStack[k]).StartPos);
                // for(int
                // i=(MyStack[k]).StartPos;i<(MyStack[k+1]).StartPos;i++)
                //	bufferOut[i+j*W] = (MyStack[k]).Value;
              }

              memset(&bufferOut[(MyStack[stackSize - 1]).StartPos + j * W],
                     (MyStack[stackSize - 1]).Value,
                     rp - (MyStack[stackSize - 1]).StartPos);
              // for(int i=(MyStack[stackSize-1]).StartPos;i<rp;i++)
              //	bufferOut[i+j*W] = (MyStack[stackSize-1]).Value;
              //	OK=1;
              break;
            }

            stackSize -= 1;
          }
        }
        // if(!OK)
        // La ligne ne respecte pas le crit�re:
        // memset(&bufferOut[(MyStack[0]).StartPos+j*W],(MyStack[0]).Value,rp-(MyStack[0]).StartPos);
        // for(int i=(MyStack[0]).StartPos;i<rp;i++)
        //	bufferOut[i+j*W] = (MyStack[0]).Value;
      }

      return RES_OK;
    }
#endif

} // namespace smil

#endif
