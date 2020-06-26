#ifndef __FAST_GRANULO_T_HPP__
#define __FAST_GRANULO_T_HPP__

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/morpheeImage.hpp>

// Vincent Morard
// 11 octobre : granulo in linear time

namespace morphee
{
  namespace FastLine
  {
    int NbPixelProcess;

    template <class T1, class T1bis>
    void BuildGranuloFromPoint(T1 F, Node<T1bis> *MyStack, int *stackSize,
                               UINT32 *granulo)
    {
      // -1-  Si la pile est vide ou si on a un front montant, on empile
      if (*stackSize == 0 || F > MyStack[*stackSize - 1].Value) {
        (MyStack[*stackSize]).StartPos = wp;
        (MyStack[*stackSize]).Passed   = 0;
        (MyStack[*stackSize]).Value    = (T1bis) F;
        *stackSize += 1;
      } else {
        while (F < (MyStack[*stackSize - 1]).Value) {
          //--1-- Stop criteria : the stack has only one node
          if (*stackSize == 1) {
            granulo[wp] += (MyStack[0].Value - F) * wp;
            (MyStack[0]).StartPos = wp;
            (MyStack[0]).Value    = (T1bis) F;
            break;
          }

          // On depile... then MyStack[*stackSize]-->NodeOut
          *stackSize -= 1;

          // Add the contribution of nodeOut into the granulo
          int Length = (wp - (MyStack[*stackSize]).StartPos);
          granulo[Length] += (((MyStack[*stackSize]).Value -
                               MAX(F, (MyStack[*stackSize - 1]).Value)) *
                              Length);

          //--2-- Stop criteria : the current pixel is bigger than the previous
          if (F > (MyStack[*stackSize - 1]).Value) { // we push it
            (MyStack[*stackSize]).Value = (T1bis) F;
            *stackSize += 1;
            break;
          }
        }
      }
      wp++;
    }

    template <class T1, typename T1bis>
    int ComputeLinePosDiag_Granulo(T1 *bufferIn, int W, int H, int x, int y,
                                   Node<T1bis> *MyStack, int *stackSize,
                                   UINT32 *granulo)
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
        BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);
        idx += W + 1;
        // p++;
        y++;
      }
      return (x - x0);
    }

    template <class T1, typename T1bis>
    int ComputeLineNegDiag_Granulo(T1 *bufferIn, int W, int H, int x, int y,
                                   Node<T1bis> *MyStack, int *stackSize,
                                   UINT32 *granulo)
    {
      int idx;
      int x0;

      if (y >= H) {
        x += y - H + 1;
        y = H - 1;
      }
      if (x >= W)
        return (-2);
      x0  = x;
      idx = y * W + x;

      BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);
      while ((x < W - 1) && (y > 0)) {
        // p++;
        x++;
        y--;
        idx -= W - 1;

        BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);
      }
      return (x - x0 + 1);
    }

    template <class T1, typename T1bis>
    int ComputeBresenhamLinePX_Granulo(T1 *bufferIn, int W, int H, int x, int y,
                                       int dx, int dy, Node<T1bis> *MyStack,
                                       int *stackSize, UINT32 *granulo)
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
        BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);

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

    template <class T1, typename T1bis>
    int ComputeBresenhamLineNX_Granulo(T1 *bufferIn, int W, int H, int x, int y,
                                       int dx, int dy, Node<T1bis> *MyStack,
                                       int *stackSize, UINT32 *granulo)
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
        return (-2);
      x0  = x;
      idx = y * W + x;

      BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);

      while ((x < W - 1) && (y > 0)) {
        if (dp >= 0) {
          y--;
          idx -= W;
          dp += twodydx;
        } else
          dp += twody;
        x++;
        idx++;

        BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);
      }
      return (x - x0 + 1);
    }

    template <class T1, typename T1bis>
    int ComputeBresenhamLinePY_Granulo(T1 *bufferIn, int W, int H, int x, int y,
                                       int dx, int dy, Node<T1bis> *MyStack,
                                       int *stackSize, UINT32 *granulo)
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
        BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);
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

    template <class T1, typename T1bis>
    int ComputeBresenhamLineNY_Granulo(T1 *bufferIn, int W, int H, int x, int y,
                                       int dx, int dy, Node<T1bis> *MyStack,
                                       int *stackSize, UINT32 *granulo)
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

      BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);

      while ((y < H - 1) && (x > 0) && (x < W)) {
        if (dp >= 0) {
          x--;
          idx--;
          dp += twodxdy;
        } else
          dp += twodx;
        y++;
        idx += W;

        BuildGranuloFromPoint(bufferIn[idx], MyStack, stackSize, granulo);
      }
      return (y - y0 + 1);
    }

    template <typename T1bis>
    void EndProcess_Granulo(Node<T1bis> *MyStack, int *stackSize,
                            UINT32 *granulo)
    {
      BuildGranuloFromPoint(0, MyStack, stackSize, granulo);
      *stackSize = 0;
    }

    template <class T1>
    void LineGranuloH(T1 *bufferIn, int W, int H, UINT32 *granulo)
    {
      // Pas génant de faire <T2> puisque T1 est du même type que T2 mais T1 est
      // const...
      Node<UINT8> MyStack[256];
      int stackPos = 0;
      int i, j;
      NbPixelProcess = W * H;
      for (j = 0; j < H; j++) // Pour toutes les lignes
      {
        wp = 0;
        for (i = 0; i < W; i++) // Pour toutes les colonnes
          BuildGranuloFromPoint(bufferIn[i + j * W], MyStack, &stackPos,
                                granulo);
        EndProcess_Granulo(MyStack, &stackPos, granulo);
      }
    }

    template <class T1>
    void LineGranuloV(T1 *bufferIn, int W, int H, UINT32 *granulo)
    {
      Node<UINT8> MyStack[256];
      int stackPos = 0;

      int i, j;
      NbPixelProcess = W * H;
      for (i = 0; i < W; i++) // Pour toutes les colonnes
      {
        wp = 0;
        for (j = 0; j < H; j++) // Pour toutes les lignes
          BuildGranuloFromPoint(bufferIn[i + j * W], MyStack, &stackPos,
                                granulo);
        EndProcess_Granulo(MyStack, &stackPos, granulo);
      }
    }

    template <class T1>
    void LineGranuloDiag(T1 *bufferIn, int W, int H, int dx, int dy,
                         UINT32 *granulo)
    {
      int y, x, nx;
      NbPixelProcess = 0;
      Node<UINT8> MyStack[256];
      int stackPos = 0;

      if (abs(dx) == abs(dy)) {
        if (dx == -dy) {
          y = 0;
          do {
            wp = 0;
            nx = ComputeLineNegDiag_Granulo(bufferIn, W, H, 0, y++, MyStack,
                                            &stackPos, granulo);
            NbPixelProcess += wp - 1;
            EndProcess_Granulo(MyStack, &stackPos, granulo);

          } while (nx > 0);

        } else {
          y = H - 1;
          do {
            wp = 0;
            nx = ComputeLinePosDiag_Granulo(bufferIn, W, H, 0, y--, MyStack,
                                            &stackPos, granulo);
            NbPixelProcess += wp - 1;
            EndProcess_Granulo(MyStack, &stackPos, granulo);
          } while (nx > 0);
        }
      } else if (abs(dx) > abs(dy)) {
        if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
          dx = abs(dx);
          dy = abs(dy);

          y = H - 1;
          do {
            wp = 0;
            nx = ComputeBresenhamLinePX_Granulo(bufferIn, W, H, 0, y--, dx, dy,
                                                MyStack, &stackPos, granulo);
            NbPixelProcess += wp - 1;
            EndProcess_Granulo(MyStack, &stackPos, granulo);
          } while (nx > 0);
        } else {
          dx = abs(dx);
          dy = abs(dy);

          y = 0;
          do {
            wp = 0;
            nx = ComputeBresenhamLineNX_Granulo(bufferIn, W, H, 0, y++, dx, dy,
                                                MyStack, &stackPos, granulo);
            NbPixelProcess += wp - 1;
            EndProcess_Granulo(MyStack, &stackPos, granulo);
          } while (nx > 0);
        }
      } else {
        if (((dx > 0) && (dy > 0)) || ((dx < 0) && (dy < 0))) {
          dx = abs(dx);
          dy = abs(dy);

          x = W - 1;
          do {
            wp = 0;
            nx = ComputeBresenhamLinePY_Granulo(bufferIn, W, H, x--, 0, dx, dy,
                                                MyStack, &stackPos, granulo);
            NbPixelProcess += wp - 1;
            EndProcess_Granulo(MyStack, &stackPos, granulo);
          } while (nx > 0);
        } else {
          dx = abs(dx);
          dy = abs(dy);

          x = 0;
          do {
            wp = 0;
            nx = ComputeBresenhamLineNY_Granulo(bufferIn, W, H, x++, 0, dx, dy,
                                                MyStack, &stackPos, granulo);
            NbPixelProcess += wp - 1;
            EndProcess_Granulo(MyStack, &stackPos, granulo);
          } while (nx > 0);
        }
      }
    }

    template <class T1>
    RES_C t_ImFastGranulo(const Image<T1> &imIn, const int angle,
                          UINT32 **granulo, int *sizeGranulo)
    {
      // Check inputs
      if (!imIn.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();

      int maxnx    = MAX(W, H);
      *sizeGranulo = 2 * maxnx;
      *granulo     = new UINT32[*sizeGranulo];
      if (*granulo == 0) {
        MORPHEE_REGISTER_ERROR("Error allocation");
        return RES_ERROR_MEMORY;
      }

      for (int i = 0; i < *sizeGranulo; i++)
        (*granulo)[i] = 0;

      int dx, dy;
      dx = (int) (cos(angle * PI / 180.0) * maxnx);
      dy = (int) (-sin(angle * PI / 180.0) * maxnx);

      if (dx == 0)
        LineGranuloV(bufferIn, W, H, *granulo);
      else if (dy == 0)
        LineGranuloH(bufferIn, W, H, *granulo);
      else
        LineGranuloDiag(bufferIn, W, H, dx, dy, *granulo);

      return RES_OK;
    }

    template <class T1, class T2>
    RES_C t_ImFastRadialGranulo(const Image<T1> &imIn, const int nbAngle,
                                const int sizeMax, Image<T2> &imOut)
    {
      // Check inputs
      if (!imIn.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();

      double *bufferOutdouble = new double[nbAngle * sizeMax];
      if (bufferOutdouble == 0) {
        MORPHEE_REGISTER_ERROR("Error allocation");
        return RES_ERROR_MEMORY;
      }
      for (int i = 0; i < nbAngle * sizeMax; i++)
        bufferOutdouble[i] = 0;

      int maxnx       = MAX(W, H);
      int sizeGranulo = 2 * maxnx;
      UINT32 *granulo = new UINT32[sizeGranulo];
      if (granulo == 0) {
        MORPHEE_REGISTER_ERROR("Error allocation");
        return RES_ERROR_MEMORY;
      }

      double maxValue = 0;

      for (int iter = 0; iter < nbAngle; iter++) {
        for (int i = 0; i < sizeGranulo; i++)
          granulo[i] = 0;

        int dx, dy, angle;
        angle = (int) (iter * 180 / (double) nbAngle);
        dx    = (int) (cos(angle * PI / 180.0) * maxnx);
        dy    = (int) (-sin(angle * PI / 180.0) * maxnx);

        if (dx == 0)
          LineGranuloV(bufferIn, W, H, granulo);
        else if (dy == 0)
          LineGranuloH(bufferIn, W, H, granulo);
        else
          LineGranuloDiag(bufferIn, W, H, dx, dy, granulo);

        for (int i = 0; i < MIN(sizeMax, sizeGranulo); i++) {
          bufferOutdouble[i + iter * sizeMax] =
              granulo[i] / (double) NbPixelProcess;
          if (maxValue < (granulo[i] / (double) NbPixelProcess))
            maxValue = granulo[i] / (double) NbPixelProcess;
        }
      }

      // Correction 1, on soustrait l'histogramme du fond de l'image (moyenne)
      if (1) {
        int Mean = 0;
        for (int i = W * H - 1; i >= 0; i--)
          Mean += bufferIn[i];
        Mean /= W * H;

        if (Mean != 0) {
          UINT8 *bufferInMean = new UINT8[W * H];
          for (int i = W * H - 1; i >= 0; i--)
            bufferInMean[i] = Mean;

          for (int iter = 0; iter < nbAngle; iter++) {
            for (int i = 0; i < sizeGranulo; i++)
              granulo[i] = 0;

            int dx, dy, angle;
            angle = (int) (iter * 180 / (double) nbAngle);
            dx    = (int) (cos(angle * PI / 180.0) * maxnx);
            dy    = (int) (-sin(angle * PI / 180.0) * maxnx);

            if (dx == 0)
              LineGranuloV(bufferInMean, W, H, granulo);
            else if (dy == 0)
              LineGranuloH(bufferInMean, W, H, granulo);
            else
              LineGranuloDiag(bufferInMean, W, H, dx, dy, granulo);

            for (int i = 0; i < MIN(sizeMax, sizeGranulo); i++) {
              bufferOutdouble[i + iter * sizeMax] -=
                  (granulo[i] / (double) NbPixelProcess);
              if (bufferOutdouble[i + iter * sizeMax] < 0)
                bufferOutdouble[i + iter * sizeMax] = 0;
            }
          }
          delete[] bufferInMean;
        }
      }

      // normalisation
      for (int i = 0; i < nbAngle * sizeMax; i++) {
        switch (imOut.getDataType()) {
        case sdtUINT8:
          bufferOut[i] = (T2)((bufferOutdouble[i]) / (double) (maxValue) *255);
          break;
        case sdtUINT16:
          bufferOut[i] =
              (T2)((bufferOutdouble[i]) / (double) (maxValue) *65535);
          break;
        case sdtUINT32:
          bufferOut[i] =
              (T2)((bufferOutdouble[i]) / (double) (maxValue) *4294967295);
          break;
        }
      }

      delete[] bufferOutdouble;
      delete[] granulo;
      return RES_OK;
    }

    struct FZ {
      int F, StartPos;
    };

    // To have a fair comparison between my algorithm and Vincent's one.
    // The interface is exactly the same!
    template <class T1>
    RES_C t_ImVincentMorardGranuloH(const Image<T1> &imIn, UINT32 **granulo,
                                    int *sizeGranulo)
    {
      // Check inputs
      if (!imIn.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();

      *sizeGranulo = W + 1;
      *granulo     = new UINT32[*sizeGranulo];
      if (*granulo == 0) {
        MORPHEE_REGISTER_ERROR("Error allocation");
        return RES_ERROR_MEMORY;
      }

      for (int i = 0; i < *sizeGranulo; i++)
        (*granulo)[i] = 0;

      int stackSize, wp, F;
      FZ MyStack[256];

      for (int j = 0; j < H; j++) {
        stackSize           = 1;
        MyStack[0].F        = bufferIn[0];
        MyStack[0].StartPos = 0;

        for (wp = 1; wp < W; wp++) {
          F = bufferIn[wp];
          if (F > MyStack[stackSize - 1].F) {
            (MyStack[stackSize]).StartPos = wp;
            (MyStack[stackSize++]).F      = F;
          } else {
            while (F < (MyStack[stackSize - 1]).F) {
              //--1-- Stop criteria : the stack has only one node
              if (stackSize == 1) {
                (*granulo)[wp] += (MyStack[0].F - F) * wp;
                (MyStack[0]).StartPos = wp;
                (MyStack[0]).F        = F;
                break;
              }

              // On depile... then MyStack[*stackSize]-->NodeOut
              stackSize--;

              // Add the contribution of nodeOut into the granulo
              int Length = (wp - (MyStack[stackSize]).StartPos);
              (*granulo)[Length] += (((MyStack[stackSize]).F -
                                      MAX(F, (MyStack[stackSize - 1]).F)) *
                                     Length);

              //--2-- Stop criteria : the current pixel is bigger than the
              //previous
              if (F > (MyStack[stackSize - 1]).F) { // we push it
                (MyStack[stackSize++]).F = F;
                break;
              }
            }
          }
        }
        // on depile les dernieres FZ de la pile
        F = 0;
        if (F > MyStack[stackSize - 1].F) {
          (MyStack[stackSize]).StartPos = wp;
          (MyStack[stackSize++]).F      = F;
        } else {
          while (F < (MyStack[stackSize - 1]).F) {
            //--1-- Stop criteria : the stack has only one node
            if (stackSize == 1) {
              (*granulo)[wp] += (MyStack[0].F - F) * wp;
              (MyStack[0]).StartPos = wp;
              (MyStack[0]).F        = F;
              break;
            }

            // On depile... then MyStack[*stackSize]-->NodeOut
            stackSize--;

            // Add the contribution of nodeOut into the granulo
            int Length = (wp - (MyStack[stackSize]).StartPos);
            (*granulo)[Length] +=
                (((MyStack[stackSize]).F - MAX(F, (MyStack[stackSize - 1]).F)) *
                 Length);

            //--2-- Stop criteria : the current pixel is bigger than the
            //previous
            if (F > (MyStack[stackSize - 1]).F) { // we push it
              (MyStack[stackSize++]).F = F;
              break;
            }
          }
        }
        bufferIn = bufferIn + W;
      }
      return RES_OK;
    }

    // To have a fair comparison between my algorithm and Vincent's one.
    // The interface is exactly the same!
    // Luc Vincent's algorithm.Paper Granulometries and opening trees
    template <class T1>
    RES_C t_ImLucVincentGranuloH(const Image<T1> &imIn, UINT32 **granulo,
                                 int *sizeGranulo)
    {
      // Check inputs
      if (!imIn.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();

      *sizeGranulo = W + 1;
      *granulo     = new UINT32[*sizeGranulo];
      if (*granulo == 0) {
        MORPHEE_REGISTER_ERROR("Error allocation");
        return RES_ERROR_MEMORY;
      }

      for (int i = 0; i < *sizeGranulo; i++)
        (*granulo)[i] = 0;

      int *LineJumpToLeft = new int[W];
      int left, right, PlateauMax, Pl, Pr, n;

      for (int j = 0; j < H; j++) {
        for (int i = 0; i < W; i++)
          LineJumpToLeft[i] = 1;

        left  = 0;
        right = 0;
        while (1) {
          if (right == W)
            break;

          right++;
          Pl = ((left == -1) ? (0) : (bufferIn[left]));
          Pr = ((right == W) ? (0) : (bufferIn[right]));

          if (Pr > Pl) {
            left = right;
          } else if (Pr < Pl) // This is a maxima!
          {
            PlateauMax = Pl;
            do {
              left = left - LineJumpToLeft[left];
              if (left < 0)
                break;
            } while (bufferIn[left] >= PlateauMax);
            Pl = ((left == -1) ? (0) : (bufferIn[left]));

            while (1) {
              // We add its contribution to the PS
              n = right - left - 1;
              (*granulo)[n] += n * (PlateauMax - MAX(Pl, Pr));

              if (right != W)
                LineJumpToLeft[right] = n + 1;

              if (Pr >= Pl) {
                left = right;
                break;
              }

              PlateauMax = Pl;

              do {
                left = left - LineJumpToLeft[left];
                if (left < 0)
                  break;
              } while (bufferIn[left] >= PlateauMax);
              Pl = ((left == -1) ? (0) : (bufferIn[left]));
            }
          }
        }
        bufferIn = bufferIn + W;
      }

      delete[] LineJumpToLeft;
      return RES_OK;
    }

    // code of Yan Bartovsky
    /*!
     * \brief Queue element.
     *
     *  A single data element of the Queue. Each queue is composed of apprx. SE
     * of these elements. Queue is an enhanced First In First Out memory.
     */
    struct queue_data {
      UINT8 Fval;
      int rp;
    };

    /*!
     * \brief Pair of Queue pointers.
     *
     *  The respective elements of the Queue are addressed by a pair of integer
     * pointers. Front points to the oldest element of the Queue, Back to the
     * newest element of the Queue.
     */
    struct queue_pointers {
      int front;
      int back;
    };

    // To have a fair comparison between my algorithm and Bart's one.
    // The interface is exactly the same!
    template <class T1>
    RES_C t_ImYanBartovskyGranuloH(const Image<T1> &imIn, UINT32 **granulo,
                                   int *sizeGranulo)
    {
      // Check inputs
      if (!imIn.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }

      int width, height;
      width            = imIn.getWxSize();
      height           = imIn.getWySize();
      const T1 *Fx_buf = imIn.rawPointer();

      *sizeGranulo = width + 1;
      *granulo     = new UINT32[*sizeGranulo];
      if (*granulo == 0) {
        MORPHEE_REGISTER_ERROR("Error allocation");
        return RES_ERROR_MEMORY;
      }
      int SE = *sizeGranulo;

      for (int i = 0; i < *sizeGranulo; i++)
        (*granulo)[i] = 0;

      // int SE_angle=0;
      int line, col;
      int im_col = 1, im_line = 1;
      int Fx_ptr = 0;

      UINT8 Fx, dFx;

      int Q_depth = SE;
      int back1, back2;
      queue_data *Q;
      queue_pointers *P;

      // Create Queue
      P = new queue_pointers;
      // The depth of Q of SE+1 is ok. The term below just removes some glitches
      // I have had.
      Q = new queue_data[(SE + 1) + 8 - ((SE + 1) & 0x07)];

      // Initiate image reading and writing pointers
      Fx_ptr = 0;

      // Iterate for every line of image
      for (line = 1; line <= height; line++) {
        // Initiate Queue pointers
        P->back  = 1;
        P->front = 1;

        // Iterate for every column of image
        for (col = 1; col <= width + SE - 1; col++) {
          // Set actual input value regarding borders
          if (col > width)
            Fx = 255;
          else
            Fx = Fx_buf[Fx_ptr++];

          // Eliminate useless values stored in Queue
          while (P->back != P->front) {
            if (P->back == 0) {
              if (Fx <= Q[Q_depth].Fval) {
                back1 = Q_depth;
              } else {
                break;
              }
            } else {
              if (Fx <= Q[P->back - 1].Fval) {
                back1 = P->back - 1;
              } else {
                break;
              }
            }

            back2 = (P->back < 2) ? (Q_depth - 1 + P->back) : (P->back - 2);

            if (Fx == Q[back1].Fval) {
              P->back = back1;
              break;
            } else if ((back1 != P->front && Q[back2].Fval < Q[back1].Fval) ||
                       col <= SE) {
              if (back1 == P->front) {
                // granu_distr[col-1]+=Q[back1].Fval - Fx;
                ;
              } else if (Fx <= Q[back2].Fval) {
                // if(col<=width)
                (*granulo)[col - Q[back2].rp - 1] +=
                    Q[back1].Fval - Q[back2].Fval;
                Q[back2].rp = Q[back1].rp;
              } else {
                // if(col<=width)
                (*granulo)[col - Q[back2].rp - 1] += Q[back1].Fval - Fx;
              }

              P->back = back1;
            } else
              break;
          }
          // Delete too old value out of Queue
          if (Q[P->front].rp + SE == col)
            P->front = (P->front == Q_depth) ? (0) : (P->front + 1);

          // Push new value into Queue
          Q[P->back].rp   = col;
          Q[P->back].Fval = Fx;
          P->back         = (P->back == Q_depth) ? (0) : (P->back + 1);

          // Output result value into output image
          // if (col>=SE)															//VM
          //	Fy_buf[Fy_ptr++]=Q[P->front].Fval;			//VM
        }
      }
      delete[] Q;
      delete P;
      return RES_OK;
    }

  } // namespace FastLine
} // namespace morphee

#endif
