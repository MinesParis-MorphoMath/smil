/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2019, Centre de Morphologie Mathematique
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
 *     Integrated into Smil Advanced Source Tree with some cosmetics
 *
 * __HEAD__ - Stop here !
 */

#ifndef __MORPHO_GRAPHV2_PATH_OPENING_T_HPP__
#define __MORPHO_GRAPHV2_PATH_OPENING_T_HPP__

#include "Core/include/DCore.h"
#include "include/MorphoPathOpening/mpo_utilities.h"

namespace smil
{

#define SQRT_2 1.414213562

#ifndef EPSILON
#define EPSILON 1e-10
#endif

  // **************************************************************************
  // ComputeLambda : The path to update Lambda : max path on the pixel x,y in
  // the direction Dir. Lambda : (size : W*H*8) 8 for the 8 directions
  // **************************************************************************
  inline float LambdaXY(float *Lambda, int x, int y, int W, int H, int Dir,
                         int Dim)
  {
    int ind = (x + y * W) * Dim + (Dir - 1);

    if (ind < H * W * Dim)
      return Lambda[ind];
    return 0.;
  }

  // . 4 . 2 .
  // 7       6
  // .   .   .
  // 5       8
  // . 1 . 3 .
  inline float ComputeLambda_GraphV2(float *Lambda, int W, int H, int x, int y,
                                     int Dir, int Dim = 8)
  {
    float L = 1.;
    
    switch (Dir) {
    case 1: // Vertical N to	S
      if (y - 1 >= 0) {
        float La = LambdaXY(Lambda, x, y - 1, W, H, Dir, Dim);

        if (x - 1 >= 0) {
          float Lb = LambdaXY(Lambda, x - 1, y - 1, W, H, Dir, Dim);

          if (std::abs(La) > EPSILON || std::abs(Lb) > EPSILON)
            L = (La <= Lb) ? (Lb + SQRT_2) : (La + 1);
        } else {
          L = La;
        }
      }
      break;

    case 2: // Vertical S to	N
      if (y + 1 < H) {
        float La = LambdaXY(Lambda, x, y + 1, W, H, Dir, Dim);

        if (x + 1 < W) {
          float Lb = LambdaXY(Lambda, x + 1, y + 1, W, H, Dir, Dim);

          if (std::abs(La) > EPSILON || std::abs(Lb) > EPSILON)
            L = (La <= Lb) ? (Lb + SQRT_2) : (La + 1);
        } else {
          L = La;
        }
      }
      break;

    case 3: // Vertical N to	S	-->2
      if (y - 1 >= 0) {
        float La = LambdaXY(Lambda, x, y - 1, W, H, Dir, Dim);

        if (x + 1 < W) {
          float Lb = LambdaXY(Lambda, x + 1, y - 1, W, H, Dir, Dim);

          if (std::abs(La) > EPSILON || std::abs(Lb) > EPSILON)
            L = (La <= Lb) ? (Lb + SQRT_2) : (La + 1);
        } else {
          L = La;
        }
      }
      break;

    case 4: // Vertical S to	N	-->2
      if (y + 1 < H) {
        float La = LambdaXY(Lambda, x, y + 1, W, H, Dir, Dim);

        if (x - 1 >= 0) {
          float Lb = LambdaXY(Lambda, x - 1, y + 1, W, H, Dir, Dim);

          if (std::abs(La) > EPSILON || std::abs(Lb) > EPSILON)
            L = (La <= Lb) ? (Lb + SQRT_2) : (La + 1);
        } else {
          L = La;
        }
      }
      break;

    case 5: // Horizontal O to	E	-->1
      if (x - 1 >= 0) {
        float La = LambdaXY(Lambda, x - 1, y, W, H, Dir, Dim);

        if (y - 1 >= 0) {
          float Lb = LambdaXY(Lambda, x - 1, y - 1, W, H, Dir, Dim);

          if (std::abs(La) > EPSILON || std::abs(Lb) > EPSILON)
            L = (La <= Lb) ? (Lb + SQRT_2) : (La + 1);
        } else {
          L = La;
        }
      }
      break;

    case 6: // Horizontal E to	O	-->1
      if (x + 1 < W) {
        float La = LambdaXY(Lambda, x + 1, y, W, H, Dir, Dim);

        if (y + 1 < H) {
          float Lb = LambdaXY(Lambda, x + 1, y + 1, W, H, Dir, Dim);

          if (std::abs(La) > EPSILON || std::abs(Lb) > EPSILON)
            L = (La <= Lb) ? (Lb + SQRT_2) : (La + 1);
        } else {
          L = La;
        }
      }
      break;

    case 7: // Horizontal O to	E	-->2
      if (x - 1 >= 0) {
        float La = LambdaXY(Lambda, x - 1, y, W, H, Dir, Dim);

        if (y + 1 < H) {
          float Lb = LambdaXY(Lambda, x - 1, y + 1, W, H, Dir, Dim);

          if (std::abs(La) > EPSILON || std::abs(Lb) > EPSILON)
            L = (La <= Lb) ? (Lb + SQRT_2) : (La + 1);
        } else {
          L = La;
        }
      }
      break;

    case 8: // Horizontal E to	O	-->2
      if (x + 1 < W) {
        float La = LambdaXY(Lambda, x + 1, y, W, H, Dir, Dim);

        if (y - 1 >= 0) {
          float Lb = LambdaXY(Lambda, x + 1, y - 1, W, H, Dir, Dim);

          if (std::abs(La) > EPSILON || std::abs(Lb) > EPSILON)
            L = (La <= Lb) ? (Lb + SQRT_2) : (La + 1);
        } else {
          L = La;
        }
      }
      break;
    }

    return L;
  }

  // Type 0 ou 2
  // Fast Path update
  static void UpdateLambdaV_GraphV2(UINT8 *imIn, float *Lambda, UINT8 *IsAlreadyPush,
                             int W, int H, std::queue<int> *FIFO_Lplus,
                             std::queue<int> *FIFO_Lmoins, int k, int Type)
  {
    int i, j, j2;
    float NewLambda;

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
          NewLambda = ComputeLambda_GraphV2(Lambda, W, H, i, j, 1 + Type);

        // We push the dependant pixel (the 3 above) if Lambda changed
        if (NewLambda != Lambda[(i + j * W) * 8 + Type]) {
          Lambda[(i + j * W) * 8 + Type] = NewLambda;
          if (j + 1 < H) {
            if (IsAlreadyPush[(i + (j + 1) * W) * 8 + Type] == 0) {
              IsAlreadyPush[(i + (j + 1) * W) * 8 + Type] = 1;
              FIFO_Lplus[j + 1].push(i);
            }
            if (Type == 2) {
              if (i - 1 >= 0 &&
                  IsAlreadyPush[(i - 1 + (j + 1) * W) * 8 + 2] == 0) {
                IsAlreadyPush[(i - 1 + (j + 1) * W) * 8 + 2] = 1;
                FIFO_Lplus[j + 1].push(i - 1);
              }
            } else {
              if (i + 1 < W && IsAlreadyPush[(i + 1 + (j + 1) * W) * 8] == 0) {
                IsAlreadyPush[(i + 1 + (j + 1) * W) * 8] = 1;
                FIFO_Lplus[j + 1].push(i + 1);
              }
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
          NewLambda = ComputeLambda_GraphV2(Lambda, W, H, i, j2, 2 + Type);

        // We push the dependant pixel (the 3 au dessus) if Lambda changed
        if (NewLambda != Lambda[(i + j2 * W) * 8 + 1 + Type]) {
          Lambda[(i + j2 * W) * 8 + 1 + Type] = NewLambda;
          if (j2 - 1 >= 0) {
            if (IsAlreadyPush[(i + (j2 - 1) * W) * 8 + 1 + Type] == 0) {
              IsAlreadyPush[(i + (j2 - 1) * W) * 8 + 1 + Type] = 1;
              FIFO_Lmoins[j2 - 1].push(i);
            }
            if (Type == 0) {
              if (i - 1 >= 0 &&
                  IsAlreadyPush[(i - 1 + (j2 - 1) * W) * 8 + 1] == 0) {
                IsAlreadyPush[(i - 1 + (j2 - 1) * W) * 8 + 1] = 1;
                FIFO_Lmoins[j2 - 1].push(i - 1);
              }
            } else {
              if (i + 1 < W &&
                  IsAlreadyPush[(i + 1 + (j2 - 1) * W) * 8 + 3] == 0) {
                IsAlreadyPush[(i + 1 + (j2 - 1) * W) * 8 + 3] = 1;
                FIFO_Lmoins[j2 - 1].push(i + 1);
              }
            }
          }
        }
      }
    }
  }

  // Type 0 ou 2
  // Fast Path	update
  static void UpdateLambdaH_GraphV2(UINT8 *imIn, float *Lambda, UINT8 *IsAlreadyPush,
                             int W, int H, std::queue<int> *FIFO_Lplus,
                             std::queue<int> *FIFO_Lmoins, int k, int Type)
  {
    int i, j, i2;
    float NewLambda;
    // For	all	the	colums
    for (i = 0; i < W; i++) {
      // Update Lambda+
      while (!FIFO_Lplus[i].empty()) {
        j = FIFO_Lplus[i].front();
        FIFO_Lplus[i].pop();
        // This pixel desapear	at this	level, --> Lambda	is 0
        if (imIn[i + j * W] == k - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda_GraphV2(Lambda, W, H, i, j, 5 + Type);

        // We push	the	dependant	pixel	(the 2 on	the	right) if	Lambda changed
        if (NewLambda != Lambda[(i + j * W) * 8 + 4 + Type]) {
          Lambda[(i + j * W) * 8 + 4 + Type] = NewLambda;
          if (i + 1 < W) {
            if (IsAlreadyPush[(i + 1 + j * W) * 8 + 4 + Type] == 0) {
              IsAlreadyPush[(i + 1 + j * W) * 8 + 4 + Type] = 1;
              FIFO_Lplus[i + 1].push(j);
            }

            if (Type == 2) {
              if (j - 1 >= 0 &&
                  IsAlreadyPush[(i + 1 + (j - 1) * W) * 8 + 6] == 0) {
                IsAlreadyPush[(i + 1 + (j - 1) * W) * 8 + 6] = 1;
                FIFO_Lplus[i + 1].push(j - 1);
              }
            } else {
              if (j + 1 < H &&
                  IsAlreadyPush[(i + 1 + (j + 1) * W) * 8 + 4] == 0) {
                IsAlreadyPush[(i + 1 + (j + 1) * W) * 8 + 4] = 1;
                FIFO_Lplus[i + 1].push(j + 1);
              }
            }
          }
        }
      }

      i2 = W - 1 - i;
      // Compute	Lambda-
      while (!FIFO_Lmoins[i2].empty()) {
        j = FIFO_Lmoins[i2].front();
        FIFO_Lmoins[i2].pop();
        // This pixel desapear	at this	level, --> Lambda	is 0
        if (imIn[i2 + j * W] == k - 1)
          NewLambda = 0;
        else
          NewLambda = ComputeLambda_GraphV2(Lambda, W, H, i2, j, 6 + Type);

        // We push	the	dependant	pixel	(the 3 pixel on	the	left)	if Lambda
        // changed
        if (NewLambda != Lambda[(i2 + j * W) * 8 + 5 + Type]) {
          Lambda[(i2 + j * W) * 8 + 5 + Type] = NewLambda;
          if (i2 - 1 >= 0) {
            if (IsAlreadyPush[(i2 - 1 + j * W) * 8 + 5 + Type] == 0) {
              IsAlreadyPush[(i2 - 1 + j * W) * 8 + 5 + Type] = 1;
              FIFO_Lmoins[i2 - 1].push(j);
            }
            if (Type == 0) {
              if (j - 1 >= 0 &&
                  IsAlreadyPush[(i2 - 1 + (j - 1) * W) * 8 + 5] == 0) {
                IsAlreadyPush[(i2 - 1 + (j - 1) * W) * 8 + 5] = 1;
                FIFO_Lmoins[i2 - 1].push(j - 1);
              }
            } else {
              if (j + 1 < H &&
                  IsAlreadyPush[(i2 - 1 + (j + 1) * W) * 8 + 7] == 0) {
                IsAlreadyPush[(i2 - 1 + (j + 1) * W) * 8 + 7] = 1;
                FIFO_Lmoins[i2 - 1].push(j + 1);
              }
            }
          }
        }
      }
    }
  }

  template <class T1, class T2>
  RES_T ImUltimatePathOpening_GraphV2(const Image<UINT8> &imIn,
                                      Image<T1> &imTrans, Image<T2> &imInd,
                                      int stop, int lambdaAttribute)
  {
    int i, j, x, y;
    UINT32 k;
    MyPOINTS Pt;

    // **************************************
    // Check inputs
    // **************************************
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imTrans)
    ASSERT_SAME_SIZE(&imIn, &imInd)

    ImageFreezer freeze(imTrans);
    ImageFreezer freeze_2(imInd);
    fill(imTrans, T1(0));
    fill(imInd, T2(0));

    //**************************************
    // Initialisation.
    //**************************************
    int W                         = imIn.getWidth();
    int H                         = imIn.getHeight();
    std::queue<int> *PQ_L_V1Plus  = new std::queue<int>[H];
    std::queue<int> *PQ_L_V1Moins = new std::queue<int>[H];
    std::queue<int> *PQ_L_V2Plus  = new std::queue<int>[H];
    std::queue<int> *PQ_L_V2Moins = new std::queue<int>[H];
    std::queue<int> *PQ_L_H1Plus  = new std::queue<int>[W];
    std::queue<int> *PQ_L_H1Moins = new std::queue<int>[W];
    std::queue<int> *PQ_L_H2Plus  = new std::queue<int>[W];
    std::queue<int> *PQ_L_H2Moins = new std::queue<int>[W];

    // each pixel have vector of MyPoints to store each changement in the PO
    // We are able to build all the Lenght for the PathOP
    std::vector<std::vector<MyPOINTS>> PathOp(W * H);

    float *Lambda = new float[W * H * 8];
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
    typename Image<T1>::lineType pixelsTrans = imTrans.getPixels();
    typename Image<T2>::lineType pixelsInd   = imInd.getPixels();

    //********************************************************
    // Initialisation:	remplir	les	structures avec	le seuil a 0
    //********************************************************
    for (i = 0; i < W; i++) {
      PQ_L_V1Plus[0].push(i);
      PQ_L_V1Moins[H - 1].push(i);
      PQ_L_V2Plus[0].push(i);
      PQ_L_V2Moins[H - 1].push(i);
    }
    for (i = 0; i < H; i++) {
      PQ_L_H1Plus[0].push(i);
      PQ_L_H1Moins[W - 1].push(i);
      PQ_L_H2Plus[0].push(i);
      PQ_L_H2Moins[W - 1].push(i);
    }
    for (i = W * H * 8 - 1; i >= 0; i--)
      IsAlreadyPush[i] = 0;

    UpdateLambdaV_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_V1Plus,
                          PQ_L_V1Moins, 0, 0);
    UpdateLambdaV_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_V2Plus,
                          PQ_L_V2Moins, 0, 2);
    UpdateLambdaH_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_H1Plus,
                          PQ_L_H1Moins, 0, 0);
    UpdateLambdaH_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_H2Plus,
                          PQ_L_H2Moins, 0, 2);

    Pt.Dist = (int) (std::min(W, H) * SQRT_2 + std::max(W, H) - std::min(W, H));
    Pt.Seuil = 0;

    for (size_t i = 0; i < imIn.getPixelCount(); ++i) {
      x = i % W;
      y = i / W;
      PathOp[i].push_back(Pt);

      // Level	k=0
      if (pixelsIn[i] == 0) {
        for (j = 0; j < 8; j++)
          IsAlreadyPush[8 * i + j] = 1;

        PQ_L_V1Plus[y].push(x);
        PQ_L_V1Moins[y].push(x);
        PQ_L_V2Plus[y].push(x);
        PQ_L_V2Moins[y].push(x);

        PQ_L_H1Plus[x].push(y);
        PQ_L_H1Moins[x].push(y);
        PQ_L_H2Plus[x].push(y);
        PQ_L_H2Moins[x].push(y);
      } else {
        for (j = 0; j < 8; j++)
          IsAlreadyPush[8 * i + j] = 0;
      }
    }

    //**********************
    // Start	of the algorithm
    //**********************
    for (k = 1; k <= 256; k++) {
      // Update Lambda	from previous	Lambda
      UpdateLambdaV_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_V1Plus,
                            PQ_L_V1Moins, k, 0);
      UpdateLambdaV_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_V2Plus,
                            PQ_L_V2Moins, k, 2);
      UpdateLambdaH_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_H1Plus,
                            PQ_L_H1Moins, k, 0);
      UpdateLambdaH_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_H2Plus,
                            PQ_L_H2Moins, k, 2);

      for (i = 0; i < W * H; i++) {
        // Mise a jour de IsAlreadyPush
        IsAlreadyPush[i * 8] = ((pixelsIn[i] <= k) ? (1) : (0));
        if (IsAlreadyPush[i * 8 + 1] != 2)
          for (int ind = 1; ind < 8; ind++)
            IsAlreadyPush[i * 8 + ind] = IsAlreadyPush[i * 8];

        // Update the FIFO with the new pixel who desapear
        if (pixelsIn[i] == k) {
          PQ_L_V1Plus[i / W].push(i % W);
          PQ_L_V1Moins[i / W].push(i % W);
          PQ_L_V2Plus[i / W].push(i % W);
          PQ_L_V2Moins[i / W].push(i % W);
          PQ_L_H1Plus[i % W].push(i / W);
          PQ_L_H1Moins[i % W].push(i / W);
          PQ_L_H2Plus[i % W].push(i / W);
          PQ_L_H2Moins[i % W].push(i / W);
        }

        // Compute the distance value which is the MAX of the path
        float Dist = std::max(
            Lambda[i * 8] + Lambda[i * 8 + 1] - 1,
            std::max(Lambda[i * 8 + 2] + Lambda[i * 8 + 3] - 1,
                     std::max(Lambda[i * 8 + 4] + Lambda[i * 8 + 5] - 1,
                              Lambda[i * 8 + 6] + Lambda[i * 8 + 7] - 1)));

        if (Dist < 0)
          Dist = 0;
        if (Dist == 0)
          IsAlreadyPush[i * 8 + 1] = 2;

        Pt = PathOp[i].back();
        if (std::abs(Pt.Dist - Dist) > EPSILON) {
          Pt.Dist  = (int) Dist;
          Pt.Seuil = k - 1;
          if (Pt.Dist == -1)
            Dist = Dist;
          PathOp[i].push_back(Pt);
        }
      }
    }

    delete[] PQ_L_H2Moins;
    delete[] PQ_L_H1Moins;
    delete[] PQ_L_H2Plus;
    delete[] PQ_L_H1Plus;
    delete[] PQ_L_V1Moins;
    delete[] PQ_L_V2Moins;
    delete[] PQ_L_V1Plus;
    delete[] PQ_L_V2Plus;
    delete[] IsAlreadyPush;
    delete[] Lambda;

    // UO initialisation
    Image<UINT8> imPOold(imIn);
    copy(imIn, imPOold);

    typename Image<UINT8>::lineType pixelsPOold = imPOold.getPixels();

    LONG ValPO, Sub;

    INT8 *Accumulation = new INT8[W * H];
    for (i = 0; i < W * H; i++)
      Accumulation[i] = 0;

    // if the Stop criteria is not defined we set it to max(W,H) -1
    if (stop < 0)
      stop = std::max(W, H) - 1;
    if (stop > std::max(W, H) - 1)
      stop = std::max(W, H) - 1;

    for (int Lenght = 1; Lenght < stop; Lenght++) {
      for (size_t i = 0; i < imTrans.getPixelCount(); ++i) {
        // Calcul de la valeur du PathOp pour la distance Lambda pour le pixel i
        for (k = 0;; k++)
          if (PathOp[i][k].Dist < Lenght) {
            ValPO = PathOp[i][k].Seuil;
            break;
          }

        if (Lenght == 1) { // Initialisation a 0
          pixelsInd[i]   = 0;
          pixelsTrans[i] = 0;
        }

        // On fait la soustraction entre le PO lambda-1 et le PO Lambda
        Sub = pixelsPOold[i] - ValPO;

        // On ecrit si on a un residu plus grand que l'ancien  (max)
        if (pixelsTrans[i] <= (ULONG) Sub && Sub > 0 && Accumulation[i] <= 0) {
          pixelsTrans[i]  = Sub;
          pixelsInd[i]    = Lenght + 1; // On ecrit l'indicatrice
          Accumulation[i] = lambdaAttribute;
        } else if (Accumulation[i] >= 1) {
          // Ou si l'accumulation est active
          pixelsTrans[i] += Sub;
          // On ecrit l'indicatrice
          pixelsInd[i] = Lenght + 1;
        }
        if (Sub == 0)
          Accumulation[i]--;

        pixelsPOold[i] = ValPO;
      }
    }
    delete[] Accumulation;

    return RES_OK;
  }

  template <class T>
  RES_T ImPathOpening_GraphV2(const Image<UINT8> &imIn, double Lenght,
                              Image<T> &imOut)
  {
    int i, j, x, y;
    UINT32 k;

    //**************************************
    // Check inputs
    //**************************************
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T(0));
    //**************************************
    // Initialisation.
    //**************************************
    int W                         = imIn.getWidth();
    int H                         = imIn.getHeight();
    std::queue<int> *PQ_L_V1Plus  = new std::queue<int>[H];
    std::queue<int> *PQ_L_V1Moins = new std::queue<int>[H];
    std::queue<int> *PQ_L_V2Plus  = new std::queue<int>[H];
    std::queue<int> *PQ_L_V2Moins = new std::queue<int>[H];
    std::queue<int> *PQ_L_H1Plus  = new std::queue<int>[W];
    std::queue<int> *PQ_L_H1Moins = new std::queue<int>[W];
    std::queue<int> *PQ_L_H2Plus  = new std::queue<int>[W];
    std::queue<int> *PQ_L_H2Moins = new std::queue<int>[W];

    float *Lambda = new float[W * H * 8];
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

    //********************************************************
    // Initialisation:	remplir	les	structures avec	le seuil a 0
    //********************************************************
    for (i = 0; i < W; i++) {
      PQ_L_V1Plus[0].push(i);
      PQ_L_V1Moins[H - 1].push(i);
      PQ_L_V2Plus[0].push(i);
      PQ_L_V2Moins[H - 1].push(i);
    }
    for (i = 0; i < H; i++) {
      PQ_L_H1Plus[0].push(i);
      PQ_L_H1Moins[W - 1].push(i);
      PQ_L_H2Plus[0].push(i);
      PQ_L_H2Moins[W - 1].push(i);
    }
    for (i = W * H * 8 - 1; i >= 0; i--)
      IsAlreadyPush[i] = 0;

    UpdateLambdaV_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_V1Plus,
                          PQ_L_V1Moins, 0, 0);
    UpdateLambdaV_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_V2Plus,
                          PQ_L_V2Moins, 0, 2);
    UpdateLambdaH_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_H1Plus,
                          PQ_L_H1Moins, 0, 0);
    UpdateLambdaH_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_H2Plus,
                          PQ_L_H2Moins, 0, 2);

    for (size_t i = 0; i < imIn.getPixelCount(); ++i) {
      x = i % W;
      y = i / W;

      // Level	k=0
      if (pixelsIn[i] == 0) {
        for (j = 0; j < 8; j++)
          IsAlreadyPush[8 * i + j] = 1;

        PQ_L_V1Plus[y].push(x);
        PQ_L_V1Moins[y].push(x);
        PQ_L_V2Plus[y].push(x);
        PQ_L_V2Moins[y].push(x);

        PQ_L_H1Plus[x].push(y);
        PQ_L_H1Moins[x].push(y);
        PQ_L_H2Plus[x].push(y);
        PQ_L_H2Moins[x].push(y);
      } else {
        for (j = 0; j < 8; j++)
          IsAlreadyPush[8 * i + j] = 0;
      }
    }

    //**********************
    // Start	of the algorithm
    //**********************
    for (k = 1; k <= 256; k++) {
      // Update Lambda	from previous	Lambda
      UpdateLambdaV_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_V1Plus,
                            PQ_L_V1Moins, k, 0);
      UpdateLambdaV_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_V2Plus,
                            PQ_L_V2Moins, k, 2);
      UpdateLambdaH_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_H1Plus,
                            PQ_L_H1Moins, k, 0);
      UpdateLambdaH_GraphV2(pixelsIn, Lambda, IsAlreadyPush, W, H, PQ_L_H2Plus,
                            PQ_L_H2Moins, k, 2);

      for (size_t i = 0; i < imOut.getPixelCount(); ++i) {
        // Mise a jour de IsAlreadyPush
        IsAlreadyPush[i * 8] = ((pixelsIn[i] <= k) ? (1) : (0));
        if (IsAlreadyPush[i * 8 + 1] != 2)
          for (int ind = 1; ind < 8; ind++)
            IsAlreadyPush[i * 8 + ind] = IsAlreadyPush[i * 8];

        // Update the FIFO with the new pixel who desapear
        if (pixelsIn[i] == k) {
          PQ_L_V1Plus[i / W].push(i % W);
          PQ_L_V1Moins[i / W].push(i % W);
          PQ_L_V2Plus[i / W].push(i % W);
          PQ_L_V2Moins[i / W].push(i % W);
          PQ_L_H1Plus[i % W].push(i / W);
          PQ_L_H1Moins[i % W].push(i / W);
          PQ_L_H2Plus[i % W].push(i / W);
          PQ_L_H2Moins[i % W].push(i / W);
        }

        // Compute the distance value which is the MAX of the path
        float Dist = std::max(
            Lambda[i * 8] + Lambda[i * 8 + 1] - 1,
            std::max(Lambda[i * 8 + 2] + Lambda[i * 8 + 3] - 1,
                     std::max(Lambda[i * 8 + 4] + Lambda[i * 8 + 5] - 1,
                              Lambda[i * 8 + 6] + Lambda[i * 8 + 7] - 1)));

        if (Dist >= Lenght || k == 1)
          pixelsOut[i] = k - 1;
      }
    }

    delete[] PQ_L_H2Moins;
    delete[] PQ_L_H1Moins;
    delete[] PQ_L_H2Plus;
    delete[] PQ_L_H1Plus;
    delete[] PQ_L_V1Moins;
    delete[] PQ_L_V2Moins;
    delete[] PQ_L_V1Plus;
    delete[] PQ_L_V2Plus;
    delete[] IsAlreadyPush;
    delete[] Lambda;

    return RES_OK;
  }
} // namespace smil

#endif
