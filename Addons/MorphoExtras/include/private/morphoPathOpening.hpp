/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2020, Centre de Morphologie Mathematique
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   Portage de MorphM vers Smil
 *
 * History :
 *   - XX/XX/XXXX - by Andres Serna
 *     Ported from MorphM
 *   - 10/02/2019 - by Jose-Marcio
 *     Integrated into Smil Advanced Source Tree with some cosmeticsg
 *
 * __HEAD__ - Stop here !
 */


#ifndef __MORPHO_PATH_OPENING_T_HPP__
#define __MORPHO_PATH_OPENING_T_HPP__

#include "Core/include/DCore.h"
#include "include/MorphoPathOpening/mpo_utilities.h"

namespace smil
{
  //*****************************************************************************
  // ComputeLambda : The path to update Lambda : max path on the pixel x,y in
  // the direction Dir. Lambda : (size : W*H*8) 8 for the 9 directions
  //*****************************************************************************
  int ComputeLambda(UINT32 *Lambda, int W, int H, int x, int y, int Dir,
                    int Dim = 8)
  {
    UINT32 L = 0;
    switch (Dir) {
    case 1: // Vertical N to S
      if (y - 1 >= 0) {
        L = Lambda[(x + (y - 1) * W) * Dim];
        if (x - 1 >= 0 && L < Lambda[(x - 1 + (y - 1) * W) * Dim])
          L = Lambda[(x - 1 + (y - 1) * W) * Dim];
        if (x + 1 < W && L < Lambda[(x + 1 + (y - 1) * W) * Dim])
          L = Lambda[(x + 1 + (y - 1) * W) * Dim];
      }
      break;

    case 2: // Vertical S to N
      if (y + 1 < H) {
        L = Lambda[(x + (y + 1) * W) * Dim + 1];
        if (x - 1 >= 0 && L < Lambda[(x - 1 + (y + 1) * W) * Dim + 1])
          L = Lambda[(x - 1 + (y + 1) * W) * Dim + 1];
        if (x + 1 < W && L < Lambda[(x + 1 + (y + 1) * W) * Dim + 1])
          L = Lambda[(x + 1 + (y + 1) * W) * Dim + 1];
      }
      break;

    case 3: // Horizontal O to E
      if (x - 1 >= 0) {
        L = Lambda[(x - 1 + y * W) * Dim + 2];
        if (y - 1 >= 0 && L < Lambda[(x - 1 + (y - 1) * W) * Dim + 2])
          L = Lambda[(x - 1 + (y - 1) * W) * Dim + 2];
        if (y + 1 < H && L < Lambda[(x - 1 + (y + 1) * W) * Dim + 2])
          L = Lambda[(x - 1 + (y + 1) * W) * Dim + 2];
      }
      break;

    case 4: // Horizontal E to O
      if (x + 1 < W) {
        L = Lambda[(x + 1 + y * W) * Dim + 3];
        if (y - 1 >= 0 && L < Lambda[(x + 1 + (y - 1) * W) * Dim + 3])
          L = Lambda[(x + 1 + (y - 1) * W) * Dim + 3];
        if (y + 1 < H && L < Lambda[(x + 1 + (y + 1) * W) * Dim + 3])
          L = Lambda[(x + 1 + (y + 1) * W) * Dim + 3];
      }
      break;

    case 5: // Diagonal SO to NE
      L = 0;
      if (y - 1 >= 0 && L < Lambda[(x + (y - 1) * W) * Dim + 4])
        L = Lambda[(x + (y - 1) * W) * Dim + 4];
      if (x - 1 >= 0 && L < Lambda[(x - 1 + y * W) * Dim + 4])
        L = Lambda[(x - 1 + y * W) * Dim + 4];
      if (y - 1 >= 0 && x - 1 >= 0 &&
          L < Lambda[(x - 1 + (y - 1) * W) * Dim + 4])
        L = Lambda[(x - 1 + (y - 1) * W) * Dim + 4];
      break;

    case 6: // Diagonal NE to SO
      L = 0;
      if (y + 1 < H && L < Lambda[(x + (y + 1) * W) * Dim + 5])
        L = Lambda[(x + (y + 1) * W) * Dim + 5];
      if (x + 1 < W && L < Lambda[(x + 1 + y * W) * Dim + 5])
        L = Lambda[(x + 1 + y * W) * Dim + 5];
      if (y + 1 < H && x + 1 < W && L < Lambda[(x + 1 + (y + 1) * W) * Dim + 5])
        L = Lambda[(x + 1 + (y + 1) * W) * Dim + 5];
      break;

    case 7: // Diagonal NO to SE
      L = 0;
      if (x - 1 >= 0 && L < Lambda[(x - 1 + y * W) * Dim + 6])
        L = Lambda[(x - 1 + y * W) * Dim + 6];
      if (y + 1 < H && L < Lambda[(x + (y + 1) * W) * Dim + 6])
        L = Lambda[(x + (y + 1) * W) * Dim + 6];
      if (y + 1 < H && x - 1 >= 0 &&
          L < Lambda[(x - 1 + (y + 1) * W) * Dim + 6])
        L = Lambda[(x - 1 + (y + 1) * W) * Dim + 6];
      break;

    case 8: // Diagonal SE to NO
      L = 0;
      if (x + 1 < W && L < Lambda[(x + 1 + y * W) * Dim + 7])
        L = Lambda[(x + 1 + y * W) * Dim + 7];
      if (y - 1 >= 0 && L < Lambda[(x + (y - 1) * W) * Dim + 7])
        L = Lambda[(x + (y - 1) * W) * Dim + 7];
      if (y - 1 >= 0 && x + 1 < W &&
          L < Lambda[(x + 1 + (y - 1) * W) * Dim + 7])
        L = Lambda[(x + 1 + (y - 1) * W) * Dim + 7];
      break;
    }

    return L + 1;
  }

  template <class T>
  RES_T ComputeBinaryPathOpening(UINT8 *imIn, int W, int H, Image<T> &imOut)
  {
    if (W != (int) imOut.getWidth() || H != (int) imOut.getHeight())
      return RES_ERR_BAD_SIZE;

    int x[8], y[8], i, k, CurrentX = 0, CurrentY = 0, Top = 0, Left = 0;

    UINT32 *Lambda = new UINT32[W * H * 8]; // 8 directions
    if (Lambda == 0)
      return RES_ERR_BAD_ALLOCATION;

    // Initialisation
    for (i = W * H * 8 - 1; i >= 0; i--)
      Lambda[i] = 0;

    for (i = 0; i < W * H; i++) {
      // For the direction, the scanning of the image is different
      x[0] = i % W;
      x[1] = x[0];
      x[2] = i / H;
      x[3] = W - 1 - x[2];
      x[4] = CurrentX;
      x[5] = W - 1 - x[4];
      x[6] = x[4];
      x[7] = x[5];
      y[0] = i / W;
      y[1] = H - 1 - y[0];
      y[2] = i % H;
      y[3] = y[2];
      y[4] = CurrentY;
      y[5] = H - 1 - y[4];
      y[6] = H - 1 - y[4];
      y[7] = H - 1 - y[5];

      for (k = 0; k < 8; k++)
        if (imIn[x[k] + y[k] * W] > 0)
          Lambda[(x[k] + y[k] * W) * 8 + k] =
              ComputeLambda(Lambda, W, H, x[k], y[k], k + 1);

      // Compute the diagonal index
      if (CurrentY + 1 > Top && CurrentY + 1 < H)
        Top = CurrentY + 1;
      if (CurrentY == H - 1)
        Left++;
      CurrentX++;
      CurrentY--;
      if (CurrentX >= W || CurrentY < 0) {
        CurrentX = Left;
        CurrentY = Top;
      }
    }

    typename Image<T>::lineType pixelsOut = imOut.getPixels();
    for (size_t i = 0; i < imOut.getPixelCount(); i++) {
      // Select the supremum of all the path calculated
      pixelsOut[i] =
          std::max(Lambda[i * 8] + Lambda[i * 8 + 1],
                   std::max(Lambda[i * 8 + 2] + Lambda[i * 8 + 3],
                            std::max(Lambda[i * 8 + 4] + Lambda[i * 8 + 5],
                                     Lambda[i * 8 + 6] + Lambda[i * 8 + 7])));
    }

    delete[] Lambda;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImPathOpeningBruteForce(const Image<T1> &imIn, const UINT32 Lenght,
                                Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));
    // Initialisation
    Image<UINT16> imLenght(imIn);
    copy(imIn, imLenght);

    // Initialisation of the binary Buffer: imBin
    int W        = imIn.getWidth();
    int H        = imIn.getHeight();
    UINT8 *imBin = new UINT8[W * H];
    if (imBin == 0) {
      return RES_ERR_BAD_ALLOCATION;
    }

    /* Mask_iterator to not compile!
    typename Image<T1>::iterator itEnd;
    typename Image<UINT16>::mask_iterator it;
    for(k=0;k<256;k++){
      t_ImThreshold(imIn,static_cast<T2>(0),static_cast<T2>(k),static_cast<T2>(0),static_cast<T2>(255),imBin);
      t_BinaryPath(imBin,imMask);
      for(it=imOut.begin_masked(imMask), itEnd=imOut.end(); it!=itEnd ;++it)
        *it = k;

    }*/

    // JOE typename Image<T1>::lineType pixelsIn = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut        = imOut.getPixels();
    typename Image<UINT16>::lineType pixelsLength = imLenght.getPixels();

    // 256 a changer pour 16 bits ou 32
    // WARNING TIME COMPUTATION -> 8 bits limitation
    for (UINT32 k = 0; k < 256; k++) {
      // Threshold at the level k
      for (size_t i = 0; i < imLenght.getPixelCount(); ++i)
        imBin[i] = ((pixelsLength[i] > k) ? 255 : 0);

      ComputeBinaryPathOpening(imBin, W, H, imLenght);
      for (size_t j = 0; j < imOut.getPixelCount(); ++j) {
        if (pixelsLength[j] >= Lenght || k == 0)
          pixelsOut[j] = k;
      }
    }
    delete[] imBin;
    return RES_OK;
  }

  // Fast Path update
  void UpdateLambdaV(UINT8 *imIn, UINT32 *Lambda, UINT8 *IsAlreadyPush, int W,
                     int H, std::queue<int> *FIFO_Lplus,
                     std::queue<int> *FIFO_Lmoins, int k)
  {
    int i, j, j2;
    UINT32 NewLambda;

    // For all the line
    for (j = 0; j < H; j++) {
      // Update Lambda+
      while (!FIFO_Lplus[j].empty()) {
        i = FIFO_Lplus[j].front();
        FIFO_Lplus[j].pop();
        // This pixel desapear at this level, --> Lambda is 0
        if (imIn[i + j * W] == k - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda(Lambda, W, H, i, j, 1);

        // We push the dependant pixel (the 3 above) if Lambda changed
        if (NewLambda != Lambda[(i + j * W) * 8]) {
          Lambda[(i + j * W) * 8] = NewLambda;
          if (j + 1 < H) {
            if (IsAlreadyPush[(i + (j + 1) * W) * 8] == 0) {
              IsAlreadyPush[(i + (j + 1) * W) * 8] = 1;
              FIFO_Lplus[j + 1].push(i);
            }
            if (i - 1 >= 0 && IsAlreadyPush[(i - 1 + (j + 1) * W) * 8] == 0) {
              IsAlreadyPush[(i - 1 + (j + 1) * W) * 8] = 1;
              FIFO_Lplus[j + 1].push(i - 1);
            }
            if (i + 1 < W && IsAlreadyPush[(i + 1 + (j + 1) * W) * 8] == 0) {
              IsAlreadyPush[(i + 1 + (j + 1) * W) * 8] = 1;
              FIFO_Lplus[j + 1].push(i + 1);
            }
          }
        }
      }

      j2 = H - 1 - j;
      // Compute Lambda-
      while (!FIFO_Lmoins[j2].empty()) {
        i = FIFO_Lmoins[j2].front();
        FIFO_Lmoins[j2].pop();
        // This pixel desapear at this level, --> Lambda is 0
        if (imIn[i + j2 * W] == k - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda(Lambda, W, H, i, j2, 2);

        // We push the dependant pixel (the 3 au dessus) if Lambda changed
        if (NewLambda != Lambda[(i + j2 * W) * 8 + 1]) {
          Lambda[(i + j2 * W) * 8 + 1] = NewLambda;
          if (j2 - 1 >= 0) {
            if (IsAlreadyPush[(i + (j2 - 1) * W) * 8 + 1] == 0) {
              IsAlreadyPush[(i + (j2 - 1) * W) * 8 + 1] = 1;
              FIFO_Lmoins[j2 - 1].push(i);
            }
            if (i - 1 >= 0 &&
                IsAlreadyPush[(i - 1 + (j2 - 1) * W) * 8 + 1] == 0) {
              IsAlreadyPush[(i - 1 + (j2 - 1) * W) * 8 + 1] = 1;
              FIFO_Lmoins[j2 - 1].push(i - 1);
            }
            if (i + 1 < W &&
                IsAlreadyPush[(i + 1 + (j2 - 1) * W) * 8 + 1] == 0) {
              IsAlreadyPush[(i + 1 + (j2 - 1) * W) * 8 + 1] = 1;
              FIFO_Lmoins[j2 - 1].push(i + 1);
            }
          }
        }
      }
    }
  }

  // Fast Path update
  void UpdateLambdaH(UINT8 *imIn, UINT32 *Lambda, UINT8 *IsAlreadyPush, int W,
                     int H, std::queue<int> *FIFO_Lplus,
                     std::queue<int> *FIFO_Lmoins, int k)
  {
    int i, j, i2;
    UINT32 NewLambda;

    // For all the colums
    for (i = 0; i < W; i++) {
      // Update Lambda+
      while (!FIFO_Lplus[i].empty()) {
        j = FIFO_Lplus[i].front();
        FIFO_Lplus[i].pop();
        // This pixel desapear at this level, --> Lambda is 0
        if (imIn[i + j * W] == k - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda(Lambda, W, H, i, j, 3);

        // We push the dependant pixel (the 3 on the right) if Lambda changed
        if (NewLambda != Lambda[(i + j * W) * 8 + 2]) {
          Lambda[(i + j * W) * 8 + 2] = NewLambda;
          if (i + 1 < W) {
            if (IsAlreadyPush[(i + 1 + j * W) * 8 + 2] == 0) {
              IsAlreadyPush[(i + 1 + j * W) * 8 + 2] = 1;
              FIFO_Lplus[i + 1].push(j);
            }
            if (j - 1 >= 0 &&
                IsAlreadyPush[(i + 1 + (j - 1) * W) * 8 + 2] == 0) {
              IsAlreadyPush[(i + 1 + (j - 1) * W) * 8 + 2] = 1;
              FIFO_Lplus[i + 1].push(j - 1);
            }
            if (j + 1 < H &&
                IsAlreadyPush[(i + 1 + (j + 1) * W) * 8 + 2] == 0) {
              IsAlreadyPush[(i + 1 + (j + 1) * W) * 8 + 2] = 1;
              FIFO_Lplus[i + 1].push(j + 1);
            }
          }
        }
      }

      i2 = W - 1 - i;
      // Compute Lambda-
      while (!FIFO_Lmoins[i2].empty()) {
        j = FIFO_Lmoins[i2].front();
        FIFO_Lmoins[i2].pop();
        // This pixel desapear at this level, --> Lambda is 0
        if (imIn[i2 + j * W] == k - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda(Lambda, W, H, i2, j, 4);

        // We push the dependant pixel (the 3 pixel on the left) if Lambda
        // changed
        if (NewLambda != Lambda[(i2 + j * W) * 8 + 3]) {
          Lambda[(i2 + j * W) * 8 + 3] = NewLambda;
          if (i2 - 1 >= 0) {
            if (IsAlreadyPush[(i2 - 1 + j * W) * 8 + 3] == 0) {
              IsAlreadyPush[(i2 - 1 + j * W) * 8 + 3] = 1;
              FIFO_Lmoins[i2 - 1].push(j);
            }
            if (j - 1 >= 0 &&
                IsAlreadyPush[(i2 - 1 + (j - 1) * W) * 8 + 3] == 0) {
              IsAlreadyPush[(i2 - 1 + (j - 1) * W) * 8 + 3] = 1;
              FIFO_Lmoins[i2 - 1].push(j - 1);
            }
            if (j + 1 < H &&
                IsAlreadyPush[(i2 - 1 + (j + 1) * W) * 8 + 3] == 0) {
              IsAlreadyPush[(i2 - 1 + (j + 1) * W) * 8 + 3] = 1;
              FIFO_Lmoins[i2 - 1].push(j + 1);
            }
          }
        }
      }
    }
  }

  // Fast Path update
  void UpdateLambdaD1(UINT8 *imIn, UINT32 *Lambda, UINT8 *IsAlreadyPush, int W,
                      int H, std::queue<int> *FIFO_Lplus,
                      std::queue<int> *FIFO_Lmoins, int Level)
  {
    int k, k2, l, i, j;
    UINT32 NewLambda;

    // For all the colums
    for (k = 0; k < W + H - 1; k++) {
      // Update Lambda+
      while (!FIFO_Lplus[k].empty()) {
        l = FIFO_Lplus[k].front();
        FIFO_Lplus[k].pop();

        if (k < H) {
          i = 0 + l;
          j = k - l;
        } else {
          i = k - (H - 1) + l;
          j = H - 1 - l;
        }

        // This pixel desapear at this level, --> Lambda is 0
        if (imIn[i + j * W] == Level - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda(Lambda, W, H, i, j, 5);

        // We push the dependant pixel (the 3 on the right) if Lambda changed
        if (NewLambda != Lambda[(i + j * W) * 8 + 4]) {
          Lambda[(i + j * W) * 8 + 4] = NewLambda;

          if (i + 1 < W && IsAlreadyPush[(i + 1 + j * W) * 8 + 4] == 0) {
            IsAlreadyPush[(i + 1 + j * W) * 8 + 4] = 1;
            if ((i + 1) + j < H)
              FIFO_Lplus[(i + 1) + j].push(i + 1);
            else
              FIFO_Lplus[(i + 1) + j].push(H - 1 - j);
          }
          if (j + 1 < H && IsAlreadyPush[(i + (j + 1) * W) * 8 + 4] == 0) {
            IsAlreadyPush[(i + (j + 1) * W) * 8 + 4] = 1;
            if (i + (j + 1) < H)
              FIFO_Lplus[i + (j + 1)].push(i);
            else
              FIFO_Lplus[i + (j + 1)].push(H - 1 - (j + 1));
          }
          if (i + 1 < W && j + 1 < H &&
              IsAlreadyPush[(i + 1 + (j + 1) * W) * 8 + 4] == 0) {
            IsAlreadyPush[(i + 1 + (j + 1) * W) * 8 + 4] = 1;
            if ((i + 1) + (j + 1) < H)
              FIFO_Lplus[(i + 1) + (j + 1)].push(i + 1);
            else
              FIFO_Lplus[(i + 1) + (j + 1)].push(H - 1 - (j + 1));
          }
        }
      }

      k2 = H + W - 1 - k - 1;
      // Compute Lambda-
      while (!FIFO_Lmoins[k2].empty()) {
        l = FIFO_Lmoins[k2].front();
        FIFO_Lmoins[k2].pop();

        if (k2 < H) {
          i = 0 + l;
          j = k2 - l;
        } else {
          i = k2 - (H - 1) + l;
          j = H - 1 - l;
        }

        // This pixel desapear at this level, --> Lambda is 0
        if (imIn[i + j * W] == Level - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda(Lambda, W, H, i, j, 6);

        // We push the dependant pixel (the 3 pixel on the left) if Lambda
        // changed
        if (NewLambda != Lambda[(i + j * W) * 8 + 5]) {
          Lambda[(i + j * W) * 8 + 5] = NewLambda;
          if (i - 1 >= 0 && IsAlreadyPush[(i - 1 + j * W) * 8 + 5] == 0) {
            IsAlreadyPush[(i - 1 + j * W) * 8 + 5] = 1;
            if ((i - 1) + j < H)
              FIFO_Lmoins[(i - 1) + j].push(i - 1);
            else
              FIFO_Lmoins[(i - 1) + j].push(H - 1 - j);
          }
          if (j - 1 >= 0 && IsAlreadyPush[(i + (j - 1) * W) * 8 + 5] == 0) {
            IsAlreadyPush[(i + (j - 1) * W) * 8 + 5] = 1;
            if (i + (j - 1) < H)
              FIFO_Lmoins[i + (j - 1)].push(i);
            else
              FIFO_Lmoins[i + (j - 1)].push(H - 1 - (j - 1));
          }
          if (i - 1 >= 0 && j - 1 >= 0 &&
              IsAlreadyPush[(i - 1 + (j - 1) * W) * 8 + 5] == 0) {
            IsAlreadyPush[(i - 1 + (j - 1) * W) * 8 + 5] = 1;
            if ((i - 1) + (j - 1) < H)
              FIFO_Lmoins[(i - 1) + (j - 1)].push(i - 1);
            else
              FIFO_Lmoins[(i - 1) + (j - 1)].push(H - 1 - (j - 1));
          }
        }
      }
    }
  }

  // Fast Path update
  void UpdateLambdaD2(UINT8 *imIn, UINT32 *Lambda, UINT8 *IsAlreadyPush, int W,
                      int H, std::queue<int> *FIFO_Lplus,
                      std::queue<int> *FIFO_Lmoins, int Level)
  {
    int k, k2, l, i, j;
    UINT32 NewLambda;

    // For all the colums
    for (k = 0; k < W + H - 1; k++) {
      // Update Lambda+
      while (!FIFO_Lplus[k].empty()) {
        l = FIFO_Lplus[k].front();
        FIFO_Lplus[k].pop();

        if (k + 1 < H) {
          i = W - 1 - l;
          j = k - l;
        } else {
          i = W - 1 + H - 1 - l - k;
          j = H - 1 - l;
        }

        // This pixel desapear at this level, --> Lambda is 0
        if (imIn[i + j * W] == Level - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda(Lambda, W, H, i, j, 8);

        // We push the dependant pixel if Lambda has changed
        if (NewLambda != Lambda[(i + j * W) * 8 + 7]) {
          Lambda[(i + j * W) * 8 + 7] = NewLambda;
          if (i - 1 >= 0 && IsAlreadyPush[(i - 1 + j * W) * 8 + 7] == 0) {
            IsAlreadyPush[(i - 1 + j * W) * 8 + 7] = 1;
            if (W - (i - 1) + j < H)
              FIFO_Lplus[W - 1 - (i - 1) + j].push(W - 1 - (i - 1));
            else
              FIFO_Lplus[W - 1 - (i - 1) + j].push(H - 1 - j);
          }
          if (j + 1 < H && IsAlreadyPush[(i + (j + 1) * W) * 8 + 7] == 0) {
            IsAlreadyPush[(i + (j + 1) * W) * 8 + 7] = 1;
            if (W - i + (j + 1) < H)
              FIFO_Lplus[W - 1 - i + (j + 1)].push(W - 1 - i);
            else
              FIFO_Lplus[W - 1 - i + (j + 1)].push(H - 1 - (j + 1));
          }
          if (i - 1 >= 0 && j + 1 < H &&
              IsAlreadyPush[(i - 1 + (j + 1) * W) * 8 + 7] == 0) {
            IsAlreadyPush[(i - 1 + (j + 1) * W) * 8 + 7] = 1;
            if (W - (i - 1) + (j + 1) < H)
              FIFO_Lplus[W - 1 - (i - 1) + (j + 1)].push(W - 1 - (i - 1));
            else
              FIFO_Lplus[W - 1 - (i - 1) + (j + 1)].push(H - 1 - (j + 1));
          }
        }
      }

      k2 = H + W - 1 - k - 1;
      // Compute Lambda-
      while (!FIFO_Lmoins[k2].empty()) {
        l = FIFO_Lmoins[k2].front();
        FIFO_Lmoins[k2].pop();

        if (k2 + 1 < H) {
          i = W - 1 - l;
          j = k2 - l;
        } else {
          i = W - 1 + H - 1 - l - k2;
          j = H - 1 - l;
        }

        // This pixel desapear at this level, --> Lambda is 0
        if (imIn[i + j * W] == Level - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda(Lambda, W, H, i, j, 7);

        // We push the dependant pixel (the 3 pixel on the left) if Lambda
        // changed
        if (NewLambda != Lambda[(i + j * W) * 8 + 6]) {
          Lambda[(i + j * W) * 8 + 6] = NewLambda;

          if (i + 1 < W && IsAlreadyPush[(i + 1 + j * W) * 8 + 6] == 0) {
            IsAlreadyPush[(i + 1 + j * W) * 8 + 6] = 1;
            if (W - (i + 1) + j < H)
              FIFO_Lmoins[W - 1 - (i + 1) + j].push(W - 1 - (i + 1));
            else
              FIFO_Lmoins[W - 1 - (i + 1) + j].push(H - 1 - j);
          }
          if (j - 1 >= 0 && IsAlreadyPush[(i + (j - 1) * W) * 8 + 6] == 0) {
            IsAlreadyPush[(i + (j - 1) * W) * 8 + 6] = 1;
            if (W - i + (j - 1) < H)
              FIFO_Lmoins[W - 1 - i + (j - 1)].push(W - 1 - i);
            else
              FIFO_Lmoins[W - 1 - i + (j - 1)].push(H - 1 - (j - 1));
          }
          if (i + 1 < W && j - 1 >= 0 &&
              IsAlreadyPush[(i + 1 + (j - 1) * W) * 8 + 6] == 0) {
            IsAlreadyPush[(i + 1 + (j - 1) * W) * 8 + 6] = 1;
            if (W - (i + 1) + (j - 1) < H)
              FIFO_Lmoins[W - 1 - (i + 1) + (j - 1)].push(W - 1 - (i + 1));
            else
              FIFO_Lmoins[W - 1 - (i + 1) + (j - 1)].push(H - 1 - (j - 1));
          }
        }
      }
    }
  }

  template <class T>
  RES_T ImPathOpening(const Image<UINT8> &imIn, const UINT32 Lenght,
                      Image<T> &imOut)
  {
    ULONG x, y;

    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T(0));

    //**********************
    // Initialisation.
    //**********************
    ULONG W                                    = imIn.getWidth();
    ULONG H                                    = imIn.getHeight();
    std::queue<int> *priorityQueueLambdaVPlus  = new std::queue<int>[H];
    std::queue<int> *priorityQueueLambdaVMoins = new std::queue<int>[H];
    std::queue<int> *priorityQueueLambdaHPlus  = new std::queue<int>[W];
    std::queue<int> *priorityQueueLambdaHMoins = new std::queue<int>[W];
    std::queue<int> *priorityQueueLambdaD1Plus = new std::queue<int>[W + H - 1];
    std::queue<int> *priorityQueueLambdaD1Moins =
        new std::queue<int>[W + H - 1];
    std::queue<int> *priorityQueueLambdaD2Plus = new std::queue<int>[W + H - 1];
    std::queue<int> *priorityQueueLambdaD2Moins =
        new std::queue<int>[W + H - 1];

    UINT32 *Lambda = new UINT32[W * H * 8];
    if (Lambda == NULL) {
      // MORPHEE_REGISTER_ERROR("Error alocation Lambda");
      return RES_ERR_BAD_ALLOCATION;
    }

    UINT8 *IsAlreadyPush = new UINT8[W * H * 8];
    if (IsAlreadyPush == NULL) {
      delete[] Lambda;
      // MORPHEE_REGISTER_ERROR("Error alocation IsAlreadyPush");
      return RES_ERR_BAD_ALLOCATION;
    }

    typename Image<UINT8>::lineType pixelsIn = imIn.getPixels();
    typename Image<T>::lineType pixelsOut    = imOut.getPixels();

    for (ULONG i = 0; i < imIn.getPixelCount(); ++i) {
      x = i % W;
      y = i / W;

      Lambda[(x + y * W) * 8]     = y + 1;
      Lambda[(x + y * W) * 8 + 1] = H - y;
      Lambda[(x + y * W) * 8 + 2] = x + 1;
      Lambda[(x + y * W) * 8 + 3] = W - x;

      Lambda[(x + y * W) * 8 + 4] = x + y + 1;
      Lambda[(x + y * W) * 8 + 5] = W + H - 1 - (x + y);
      Lambda[(x + y * W) * 8 + 6] = W - x + y;
      Lambda[(x + y * W) * 8 + 7] = H + x - y;

      if (pixelsIn[i] == 0) { // Level k=0
        IsAlreadyPush[8 * i]     = 1;
        IsAlreadyPush[8 * i + 1] = 1;
        IsAlreadyPush[8 * i + 2] = 1;
        IsAlreadyPush[8 * i + 3] = 1;
        IsAlreadyPush[8 * i + 4] = 1;
        IsAlreadyPush[8 * i + 5] = 1;
        IsAlreadyPush[8 * i + 6] = 1;
        IsAlreadyPush[8 * i + 7] = 1;
        priorityQueueLambdaVPlus[y].push(x);
        priorityQueueLambdaVMoins[y].push(x);
        priorityQueueLambdaHPlus[x].push(y);
        priorityQueueLambdaHMoins[x].push(y);

        if (x + y < H)
          priorityQueueLambdaD1Plus[x + y].push(x);
        else
          priorityQueueLambdaD1Plus[x + y].push(H - 1 - y);
        if (x + y < H)
          priorityQueueLambdaD1Moins[x + y].push(x);
        else
          priorityQueueLambdaD1Moins[x + y].push(H - 1 - y);
        if (W - x + y < H)
          priorityQueueLambdaD2Plus[W - 1 - x + y].push(W - 1 - x);
        else
          priorityQueueLambdaD2Plus[W - 1 - x + y].push(H - 1 - y);
        if (W - x + y < H)
          priorityQueueLambdaD2Moins[W - 1 - x + y].push(W - 1 - x);
        else
          priorityQueueLambdaD2Moins[W - 1 - x + y].push(H - 1 - y);

      } else {
        IsAlreadyPush[8 * i]     = 0;
        IsAlreadyPush[8 * i + 1] = 0;
        IsAlreadyPush[8 * i + 2] = 0;
        IsAlreadyPush[8 * i + 3] = 0;
        IsAlreadyPush[8 * i + 4] = 0;
        IsAlreadyPush[8 * i + 5] = 0;
        IsAlreadyPush[8 * i + 6] = 0;
        IsAlreadyPush[8 * i + 7] = 0;
      }
    }

    //**********************
    // Start of the algorithm
    //**********************
    for (int k = 1; k < 256; k++) {
      // Update Lambda from previous Lambda
      UpdateLambdaV(pixelsIn, Lambda, IsAlreadyPush, W, H,
                    priorityQueueLambdaVPlus, priorityQueueLambdaVMoins, k);
      UpdateLambdaH(pixelsIn, Lambda, IsAlreadyPush, W, H,
                    priorityQueueLambdaHPlus, priorityQueueLambdaHMoins, k);
      UpdateLambdaD1(pixelsIn, Lambda, IsAlreadyPush, W, H,
                     priorityQueueLambdaD1Plus, priorityQueueLambdaD1Moins, k);
      UpdateLambdaD2(pixelsIn, Lambda, IsAlreadyPush, W, H,
                     priorityQueueLambdaD2Plus, priorityQueueLambdaD2Moins, k);

      for (size_t i = 0; i < imOut.getPixelCount(); ++i) {
        // Update the mask for the pixel who have already been computed and
        // (==k) : the next ones
        IsAlreadyPush[i * 8] = ((pixelsIn[i] <= k) ? (1) : (0));
        for (int ind = 1; ind < 8; ind++)
          IsAlreadyPush[i * 8 + ind] = IsAlreadyPush[i * 8];

        // Update the FIFO with the new pixel who desapear
        if (pixelsIn[i] == k) {
          priorityQueueLambdaVPlus[i / W].push(i % W);
          priorityQueueLambdaVMoins[i / W].push(i % W);
          priorityQueueLambdaHPlus[i % W].push(i / W);
          priorityQueueLambdaHMoins[i % W].push(i / W);
          if (i / W + i % W < H)
            priorityQueueLambdaD1Plus[i / W + i % W].push(i % W);
          else
            priorityQueueLambdaD1Plus[i / W + i % W].push(H - 1 - (i / W));
          if (i / W + i % W < H)
            priorityQueueLambdaD1Moins[i / W + i % W].push(i % W);
          else
            priorityQueueLambdaD1Moins[i / W + i % W].push(H - 1 - (i / W));
          if (W - i % W + i / W < H)
            priorityQueueLambdaD2Plus[W - 1 - i % W + i / W].push(W - 1 -
                                                                  i % W);
          else
            priorityQueueLambdaD2Plus[W - 1 - i % W + i / W].push(H - 1 -
                                                                  i / W);
          if (W - i % W + i / W < H)
            priorityQueueLambdaD2Moins[W - 1 - i % W + i / W].push(W - 1 -
                                                                   i % W);
          else
            priorityQueueLambdaD2Moins[W - 1 - i % W + i / W].push(H - 1 -
                                                                   i / W);
        }

        // Compute the output with the threshold Lenght
        if (Lambda[i * 8] + Lambda[i * 8 + 1] > Lenght ||
            Lambda[i * 8 + 2] + Lambda[i * 8 + 3] > Lenght ||
            Lambda[i * 8 + 4] + Lambda[i * 8 + 5] > Lenght ||
            Lambda[i * 8 + 6] + Lambda[i * 8 + 7] > Lenght)
          pixelsOut[i] = k - 1;
      }
    }

    // clear the memory
    delete[] priorityQueueLambdaVPlus;
    delete[] priorityQueueLambdaVMoins;
    delete[] priorityQueueLambdaHPlus;
    delete[] priorityQueueLambdaHMoins;
    delete[] priorityQueueLambdaD1Plus;
    delete[] priorityQueueLambdaD1Moins;
    delete[] priorityQueueLambdaD2Plus;
    delete[] priorityQueueLambdaD2Moins;
    delete[] Lambda;
    delete[] IsAlreadyPush;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImUltimatePathOpening(const Image<UINT8> &imIn, Image<T1> &imOut,
                              Image<T2> &imIndicatrice, int stop,
                              int lambdaAttribute)
  {
    int x, y;
    MyPOINTS Pt;

    //**************************************
    // Check inputs
    //**************************************
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    ASSERT_SAME_SIZE(&imIn, &imIndicatrice)

    ImageFreezer freeze(imOut);
    ImageFreezer freeze_2(imIndicatrice);
    fill(imOut, T1(0));
    fill(imIndicatrice, T2(0));
    //**************************************
    // Initialisation.
    //**************************************
    int W                                      = imIn.getWidth();
    int H                                      = imIn.getHeight();
    std::queue<int> *priorityQueueLambdaVPlus  = new std::queue<int>[H];
    std::queue<int> *priorityQueueLambdaVMoins = new std::queue<int>[H];
    std::queue<int> *priorityQueueLambdaHPlus  = new std::queue<int>[W];
    std::queue<int> *priorityQueueLambdaHMoins = new std::queue<int>[W];

    // each pixel have vector of MyPoints to store each changement in the PO
    // We are able to build all the Lenght for the PathOP
    std::vector<std::vector<MyPOINTS> > PathOp(W * H);

    UINT32 *Lambda = new UINT32[W * H * 8];
    if (Lambda == NULL) {
      // MORPHEE_REGISTER_ERROR("Error alocation Lambda");
      return RES_ERR_BAD_ALLOCATION;
    }

    UINT8 *IsAlreadyPush = new UINT8[W * H * 8];
    if (IsAlreadyPush == NULL) {
      delete[] Lambda;
      // MORPHEE_REGISTER_ERROR("Error alocation IsAlreadyPush");
      return RES_ERR_BAD_ALLOCATION;
    }

    typename Image<UINT8>::lineType pixelsIn = imIn.getPixels();
    typename Image<T1>::lineType pixelsOut   = imOut.getPixels();
    typename Image<T2>::lineType pixelsInd   = imIndicatrice.getPixels();

    Pt.Seuil = 0;
    Pt.Dist  = W + H - 1;

    for (size_t i = 0; i < imIn.getPixelCount(); ++i) {
      x = i % W;
      y = i / W;

      Lambda[(x + y * W) * 8]     = y + 1;
      Lambda[(x + y * W) * 8 + 1] = H - y;
      Lambda[(x + y * W) * 8 + 2] = x + 1;
      Lambda[(x + y * W) * 8 + 3] = W - x;

      PathOp[i].push_back(Pt);

      // Level k=0
      if (pixelsIn[i] == 0) {
        IsAlreadyPush[8 * i]     = 1;
        IsAlreadyPush[8 * i + 1] = 1;
        IsAlreadyPush[8 * i + 2] = 1;
        IsAlreadyPush[8 * i + 3] = 1;
        IsAlreadyPush[8 * i + 4] = 1;
        IsAlreadyPush[8 * i + 5] = 1;
        IsAlreadyPush[8 * i + 6] = 1;
        IsAlreadyPush[8 * i + 7] = 1;
        priorityQueueLambdaVPlus[y].push(x);
        priorityQueueLambdaVMoins[y].push(x);
        priorityQueueLambdaHPlus[x].push(y);
        priorityQueueLambdaHMoins[x].push(y);
      }
    }

    //*********************************
    // Start of the algorithm
    //*********************************
    for (UINT32 k = 1; k <= 256; k++) {
      // Update Lambda from previous Lambda
      UpdateLambdaV(pixelsIn, Lambda, IsAlreadyPush, W, H,
                    priorityQueueLambdaVPlus, priorityQueueLambdaVMoins, k);
      UpdateLambdaH(pixelsIn, Lambda, IsAlreadyPush, W, H,
                    priorityQueueLambdaHPlus, priorityQueueLambdaHMoins, k);

      for (int i = 0; i < W * H; i++) {
        // Update the mask for the pixel who have already been computed and
        // (==k) : the next ones
        IsAlreadyPush[i * 8] = ((pixelsIn[i] <= k) ? (1) : (0));
        for (int ind = 1; ind < 8; ind++)
          IsAlreadyPush[i * 8 + ind] = IsAlreadyPush[i * 8];

        // Update the FIFO with the new pixel who desapear
        if (pixelsIn[i] == k) {
          priorityQueueLambdaVPlus[i / W].push(i % W);
          priorityQueueLambdaVMoins[i / W].push(i % W);
          priorityQueueLambdaHPlus[i % W].push(i / W);
          priorityQueueLambdaHMoins[i % W].push(i / W);
        }

        int Distance = std::max(Lambda[i * 8] + Lambda[i * 8 + 1] - 1,
                                Lambda[i * 8 + 2] + Lambda[i * 8 + 3] - 1);
        if (Distance < 0)
          Distance = 0;

        Pt = PathOp[i].back();

        if (Pt.Dist != Distance) {
          Pt.Dist  = Distance;
          Pt.Seuil = k - 1;
          PathOp[i].push_back(Pt);
        }
      }
    }

    // clear the memory for the PathOP
    delete[] priorityQueueLambdaVPlus;
    delete[] priorityQueueLambdaVMoins;
    delete[] priorityQueueLambdaHPlus;
    delete[] priorityQueueLambdaHMoins;

    delete[] Lambda;
    delete[] IsAlreadyPush;

    // UO initialisation
    Image<UINT8> imPOold(imIn);
    copy(imIn, imPOold);

    typename Image<UINT8>::lineType pixelsPo = imPOold.getPixels();

    LONG ValPO, Sub;

    INT8 *Accumulation = new INT8[W * H];
    for (int i = 0; i < W * H; i++)
      Accumulation[i] = 0;

    // if the Stop criteria is not defined we set it to max(W,H) -1
    if (stop < 0)
      stop = std::max(W, H) - 1;
    if (stop > std::max(W, H) - 1)
      stop = std::max(W, H) - 1;

    for (int Lenght = 1; Lenght < stop; Lenght++) {
      for (size_t i = 0; i < imIn.getPixelCount(); ++i) {
        // Calcul de la valeur du PathOp pour la distance Lambda pour le pixel i
        for (UINT32 k = 0;; k++)
          if (PathOp[i][k].Dist < Lenght) {
            ValPO = PathOp[i][k].Seuil;
            break;
          }

        if (Lenght == 1) { // Initialisation a 0
          pixelsInd[i] = 0;
          pixelsOut[i] = 0;
        }

        // On fait la soustraction entre le PO lambda-1 et le PO Lambda
        Sub = pixelsPo[i] - ValPO;

        if (pixelsOut[i] <= Sub && Sub > 0 && Accumulation[i] <= 0) {
          // On ecrit si on a un residu plus grand que l'ancien  (max)
          pixelsOut[i] = Sub;
          pixelsInd[i] = Lenght + 1; // On ecrit l'indicatrice
          Accumulation[i] = lambdaAttribute;
        }

        else if (Accumulation[i] >= 1) {
          // Ou si l'accumulation est active
          pixelsOut[i] += Sub;
          // On ecrit l'indicatrice
          pixelsInd[i] = Lenght + 1;
        }
        if (Sub == 0)
          Accumulation[i]--;

        pixelsPo[i] = ValPO;
      }
    }
    delete[] Accumulation;

    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImBinaryPathOpening(const Image<T1> &imIn, const UINT32 Lenght,
                            const UINT32 Slice, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));
    // Initialisation

    Image<UINT16> imLenght(imIn);
    copy(imIn, imLenght);

    // Initialisation of the binary Buffer: imBin
    int W        = imIn.getWidth();
    int H        = imIn.getHeight();
    UINT8 *imBin = new UINT8[W * H];
    if (imBin == NULL) {
      // MORPHEE_REGISTER_ERROR("Error alocation imBin");
      return RES_ERR_BAD_ALLOCATION;
    }

    typename Image<T1>::lineType pixelsIn         = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut        = imOut.getPixels();
    typename Image<UINT16>::lineType pixelsLength = imLenght.getPixels();

    // Threshold at the level "Slice"
    for (size_t i = 0; i < imIn.getPixelCount(); ++i)
      imBin[i] = ((pixelsIn[i] > Slice) ? 255 : 0);

    ComputeBinaryPathOpening(imBin, W, H, imLenght);
    for (size_t j = 0; j < imOut.getPixelCount(); ++j)
      pixelsOut[j] = ((pixelsLength[j] >= Lenght) ? (255) : (0));

    delete[] imBin;

    return RES_OK;
  }

} // namespace smil

#endif
