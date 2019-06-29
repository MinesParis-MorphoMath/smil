#ifndef __FAST_AREA_OPENIN_LINE_T_HPP__
#define __FAST_AREA_OPENIN_LINE_T_HPP__

#include "Core/include/DCore.h"

#include <queue>

// Author Vincent Morard
// Date : 7 march 2011
// See :
//	Morard V. and Dokladal P. and Decenciere E.
//	LINEAR OPENINGS IN CONSTANT TIME WITH ARBITRARY ORIENTATIONS
// This algorithm is not exact !

namespace smil
{
  struct AreaOP {
    int F, StartPos, Passed, PosInBuffer;
  };

#define NOT_DONE -1
#define IN_THE_FAH -2

  template <class T1, class T2>
  RES_T ImAreaOpening_Line(const Image<T1> &imIn, int size, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W                                  = imIn.getWidth();
    int H                                  = imIn.getHeight();
    typename Image<T1>::lineType bufferIn  = imIn.getPixels();
    typename Image<T2>::lineType bufferOut = imOut.getPixels();

    std::queue<int> FAH[256];
    AreaOP MyStack[256];
    int currentPixel, X, Y, i, j, k, lastPixel, wp, stackSize, index;
    T1 levelMaxInFAH;
    int *pixelDone = new int[W * H];

    // Initialisation
    memset(pixelDone, NOT_DONE, W * H * sizeof(int));

    pixelDone[0] = IN_THE_FAH;
    currentPixel = 0;
    lastPixel    = currentPixel;

    // Add the first pixel in the queue
    levelMaxInFAH            = bufferIn[currentPixel];
    (MyStack[0]).StartPos    = 0;
    (MyStack[0]).Passed      = 0;
    (MyStack[0]).F           = levelMaxInFAH;
    (MyStack[0]).PosInBuffer = currentPixel;
    wp                       = 1;
    stackSize                = 1;
    // JOE T1 F;
    INT F;

    while (1) {
      X = currentPixel % W;
      Y = currentPixel / W;

      // Search for neighbor
      for (k = 0; k < 4; k++) {
        switch (k) {
        case 0:
          i = X;
          j = Y - 1;
          break;
        case 1:
          i = X - 1;
          j = Y;
          break;
        case 2:
          i = X + 1;
          j = Y;
          break;
        case 3:
          i = X;
          j = Y + 1;
          break;
        }
        if (i >= 0 && i < W && j >= 0 && j < H &&
            pixelDone[i + j * W] == NOT_DONE) {
          if (levelMaxInFAH < bufferIn[i + j * W])
            levelMaxInFAH = bufferIn[i + j * W];
          FAH[bufferIn[i + j * W]].push(i + j * W);
          pixelDone[i + j * W] = IN_THE_FAH;
        }
      }

      currentPixel = -1;
      for (i = levelMaxInFAH; i >= 0; i--) // A mieux faire!!!
        if (!FAH[i].empty()) {
          currentPixel = FAH[i].front();
          FAH[i].pop();
          levelMaxInFAH = i;
          break;
        }
      // No pixel in the FAH -> End of treatment
      if (currentPixel == -1)
        break;

      //
      F = bufferIn[currentPixel];
      // -1-   si on a un front montant, on empile
      if (F > MyStack[stackSize - 1].F) { // Test *stackSize == 0  à supprimer
        (MyStack[stackSize]).StartPos    = wp;
        (MyStack[stackSize]).Passed      = 0;
        (MyStack[stackSize]).F           = F;
        (MyStack[stackSize]).PosInBuffer = currentPixel;
        stackSize++;
      } else {
        while (F < (MyStack[stackSize - 1]).F) {
          // On depile... then MyStack[*stackSize]-->NodeOut
          stackSize -= 1;

          // We have passed the criteria
          if (MyStack[stackSize].Passed ||
              (wp - (MyStack[stackSize]).StartPos >= size)) {
            // Write output
            for (int j = 0; j < stackSize; j++) {
              index = (MyStack[j]).PosInBuffer;
              for (int i = (MyStack[j]).StartPos; i < (MyStack[j + 1]).StartPos;
                   i++, index = pixelDone[index])
                bufferOut[index] = (MyStack[j]).F;
            }
            index = (MyStack[stackSize]).PosInBuffer;
            for (int i = (MyStack[stackSize]).StartPos; index >= 0;
                 i++, index = pixelDone[index])
              bufferOut[index] = (MyStack[stackSize]).F;

            (MyStack[0]).StartPos    = wp;
            (MyStack[0]).Passed      = 1;
            (MyStack[0]).F           = F;
            (MyStack[0]).PosInBuffer = currentPixel;
            stackSize                = 1;
            break;
          }
          if (stackSize == 0 || F > (MyStack[stackSize - 1]).F) {
            (MyStack[stackSize]).F = F;
            stackSize++;
            break;
          }
        }
      }
      wp++;

      //

      pixelDone[lastPixel] = currentPixel;
      lastPixel            = currentPixel;
    }

    // Il faut vider la pile!
    while (stackSize != 0) {
      if (wp - 1 - (MyStack[stackSize - 1]).StartPos >= size) {
        for (j = 0; j < stackSize - 1; j++) {
          index = (MyStack[j]).PosInBuffer;
          for (i = (MyStack[j]).StartPos; i < (MyStack[j + 1]).StartPos;
               i++, index = pixelDone[index])
            bufferOut[index] = (MyStack[j]).F;
        }
        index = (MyStack[stackSize - 1]).PosInBuffer;
        for (i = (MyStack[stackSize - 1]).StartPos; index >= 0;
             i++, index = pixelDone[index])
          bufferOut[index] = (MyStack[stackSize - 1]).F;

        delete[] pixelDone;
        return RES_OK;
      }
      stackSize--;
    }

    // La ligne ne respecte pas le critère:
    index = (MyStack[0]).PosInBuffer;
    for (int i = (MyStack[0]).StartPos; index >= 0;
         i++, index = pixelDone[index])
      bufferOut[index] = (MyStack[0]).F;

    delete[] pixelDone;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImAreaOpening_LineSupEqu(const Image<T1> &imIn, int size,
                                 Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W                                  = imIn.getWidth();
    int H                                  = imIn.getHeight();
    typename Image<T1>::lineType bufferIn  = imIn.getPixels();
    typename Image<T2>::lineType bufferOut = imOut.getPixels();

    std::queue<int> FAH[256];
    AreaOP MyStack[256];
    int currentPixel, X, Y, i, j, k, levelMaxInFAH, lastPixel, wp, stackSize,
        index;
    int *pixelDone = new int[W * H];

    // Initialisation
    memset(pixelDone, NOT_DONE, W * H * sizeof(int));

    pixelDone[0] = IN_THE_FAH;
    currentPixel = 0;
    lastPixel    = currentPixel;

    // Add the first pixel in the queue
    levelMaxInFAH            = bufferIn[currentPixel];
    (MyStack[0]).StartPos    = 0;
    (MyStack[0]).Passed      = 0;
    (MyStack[0]).F           = levelMaxInFAH;
    (MyStack[0]).PosInBuffer = currentPixel;
    wp                       = 1;
    stackSize                = 1;
    // JOE T1 F;
    INT F;

    while (1) {
      X = currentPixel % W;
      Y = currentPixel / W;

      // Search for neighbor
      for (k = 0; k < 4; k++) {
        switch (k) {
        case 0:
          i = X;
          j = Y - 1;
          break;
        case 1:
          i = X - 1;
          j = Y;
          break;
        case 2:
          i = X + 1;
          j = Y;
          break;
        case 3:
          i = X;
          j = Y + 1;
          break;
        }
        if (i >= 0 && i < W && j >= 0 && j < H &&
            pixelDone[i + j * W] == NOT_DONE) {
          FAH[bufferIn[i + j * W]].push(i + j * W);
          pixelDone[i + j * W] = IN_THE_FAH;
        }
      }

      F            = bufferIn[currentPixel];
      currentPixel = -1;
      for (i = F; i <= 255; i++) // A mieux faire!!!
        if (!FAH[i].empty()) {
          currentPixel = FAH[i].front();
          FAH[i].pop();
          break;
        }
      // No pixel in the FAH -> End of treatment
      if (currentPixel == -1) {
        for (i = F; i >= 0; i--) // A mieux faire!!!
          if (!FAH[i].empty()) {
            currentPixel = FAH[i].front();
            FAH[i].pop();
            break;
          }
        if (currentPixel == -1)
          break;
      }

      //
      F = bufferIn[currentPixel];
      // -1-   si on a un front montant, on empile
      if (F > MyStack[stackSize - 1].F) { // Test *stackSize == 0  à supprimer
        (MyStack[stackSize]).StartPos    = wp;
        (MyStack[stackSize]).Passed      = 0;
        (MyStack[stackSize]).F           = F;
        (MyStack[stackSize]).PosInBuffer = currentPixel;
        stackSize++;
      } else {
        while (F < (MyStack[stackSize - 1]).F) {
          // On depile... then MyStack[*stackSize]-->NodeOut
          stackSize -= 1;

          // We have passed the criteria
          if (MyStack[stackSize].Passed ||
              (wp - (MyStack[stackSize]).StartPos >= size)) {
            // Write output
            for (int j = 0; j < stackSize; j++) {
              index = (MyStack[j]).PosInBuffer;
              for (int i = (MyStack[j]).StartPos; i < (MyStack[j + 1]).StartPos;
                   i++, index = pixelDone[index])
                bufferOut[index] = (MyStack[j]).F;
            }
            index = (MyStack[stackSize]).PosInBuffer;
            for (int i = (MyStack[stackSize]).StartPos; index >= 0;
                 i++, index = pixelDone[index])
              bufferOut[index] = (MyStack[stackSize]).F;

            (MyStack[0]).StartPos    = wp;
            (MyStack[0]).Passed      = 1;
            (MyStack[0]).F           = F;
            (MyStack[0]).PosInBuffer = currentPixel;
            stackSize                = 1;
            break;
          }
          if (stackSize == 0 || F > (MyStack[stackSize - 1]).F) {
            (MyStack[stackSize]).F = F;
            stackSize++;
            break;
          }
        }
      }
      wp++;

      //

      pixelDone[lastPixel] = currentPixel;
      lastPixel            = currentPixel;
    }

    // Il faut vider la pile!
    while (stackSize != 0) {
      if (wp - 1 - (MyStack[stackSize - 1]).StartPos >= size) {
        for (j = 0; j < stackSize - 1; j++) {
          index = (MyStack[j]).PosInBuffer;
          for (i = (MyStack[j]).StartPos; i < (MyStack[j + 1]).StartPos;
               i++, index = pixelDone[index])
            bufferOut[index] = (MyStack[j]).F;
        }
        index = (MyStack[stackSize - 1]).PosInBuffer;
        for (i = (MyStack[stackSize - 1]).StartPos; index >= 0;
             i++, index = pixelDone[index])
          bufferOut[index] = (MyStack[stackSize - 1]).F;

        delete[] pixelDone;
        return RES_OK;
      }
      stackSize--;
    }

    // La ligne ne respecte pas le critère:
    index = (MyStack[0]).PosInBuffer;
    for (int i = (MyStack[0]).StartPos; index >= 0;
         i++, index = pixelDone[index])
      bufferOut[index] = (MyStack[0]).F;

    delete[] pixelDone;
    return RES_OK;
  }

} // namespace smil

#endif
