/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2021, Centre de Morphologie Mathematique
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

#ifndef __MORPHO_GEODESIC_PATH_OPENING_T_HPP__
#define __MORPHO_GEODESIC_PATH_OPENING_T_HPP__

#include <cmath>
#include <cstring>
#include "Core/include/DCore.h"
#include "include/MorphoPathOpening/mpo_utilities.h"

namespace smil
{
#define FLOAT_EQ_CORRIGE(x, y) (abs(x - y) < 0.000001)

  void GeodesicPathCC(double *DistanceMap, std::queue<int> *fifoSave, int W,
                      int H, int D, int BaryX, int BaryY, int BaryZ,
                      int Allongement = 0, float ScaleX = 1, float ScaleY = 1,
                      float ScaleZ = 1)
  {
    if (fifoSave->empty())
      return;

    int   IndStart, currentPixel, X, Y, Z, Xtort, Ytort, Ztort;
    float DistMax = -1, Dist;

    // Find the BaryCentre
    std::queue<int> fifoCurrent = *fifoSave; // Copy of fifoCurrent

    // Find the first Pixel (farest from the barycenter -- using Eucludean
    // distance because barycenter could not be inside the object)
    do {
      currentPixel = fifoCurrent.front();
      fifoCurrent.pop();
      X = currentPixel % W;
      Y = (currentPixel % (W * H) - X) / W;
      Z = (currentPixel - X - Y * W) / (W * H);
      // Dist = std::sqrt(pow((double)(X*ScaleX -
      // BaryX),2)+pow((double)(Y*ScaleY-BaryY),2)+pow((double)(Z*ScaleZ -
      // BaryZ),2));
      Dist = (float) std::sqrt(
          (double) ((X * ScaleX - BaryX) * (X * ScaleX - BaryX) +
                    (Y * ScaleY - BaryY) * (Y * ScaleY - BaryY) +
                    (Z * ScaleZ - BaryZ) * (Z * ScaleZ - BaryZ)));

      if (Dist > DistMax) {
        DistMax  = Dist;
        IndStart = currentPixel;
      }
    } while (!fifoCurrent.empty());

    // Get the max distance
    bool NewDist;
    int  k, l, Ind, j;
    DistanceMap[IndStart] = 1;
    fifoCurrent.push(IndStart);
    DistMax = 1;

    do {
      currentPixel = fifoCurrent.front();
      fifoCurrent.pop();
      X = currentPixel % W;
      Y = (currentPixel % (W * H) - X) / W;
      Z = (currentPixel - X - Y * W) / (W * H);

      Dist    = static_cast<float>(ImDtTypes<float>::max());
      NewDist = 0;

      // For all the neighbour
      for (j = -1; j <= 1; j++)
        if (Z + j >= 0 && Z + j < D)
          for (k = -1; k <= 1; k++)
            if (X + k >= 0 && X + k < W)
              for (l = -1; l <= 1; l++)
                if (Y + l >= 0 && Y + l < H && (k != 0 || l != 0 || j != 0)) {
                  Ind = X + k + (Y + l) * W + (Z + j) * W * H;
                  if (DistanceMap[Ind] == -2) {
                    fifoCurrent.push(Ind);
                    DistanceMap[Ind] = -3;
                  } else if (DistanceMap[Ind] != 0 && DistanceMap[Ind] != -3) {
                    float D = std::sqrt((j * ScaleX * j * ScaleX) +
                                        (k * ScaleY) * (k * ScaleY) +
                                        (l * ScaleZ) * (l * ScaleZ));
                    if (NewDist == 0 || Dist > (DistanceMap[Ind] + D)) {
                      Dist    = (float) DistanceMap[Ind] + D;
                      NewDist = 1;
                    }
                  }

                  // !(FLOAT_EQ_CORRIGE(Dist,DataTraits<F_SIMPLE
                  // !>::default_value::max_value()))){ //==> Dist !=
                  // max_value()
                  if (NewDist) {
                    DistanceMap[X + Y * W + Z * W * H] = Dist;
                    if (Dist > DistMax) {
                      DistMax = Dist;
                      Xtort   = X;
                      Ytort   = Y;
                      Ztort   = Z;
                    }
                  }
                }
    } while (!fifoCurrent.empty());

    // Write on DistanceMap the Max Distance for all the pixel of the CC, we pop
    // fifoSave
    int size = (int) fifoSave->size();
    do {
      currentPixel = fifoSave->front();
      fifoSave->pop();
      if (Allongement == 0) {
        // geodesic diameter
        DistanceMap[currentPixel] = DistMax;
        continue;
      }
      if (Allongement == 1) {
        if (D == 1) {
          // En 2D
          DistanceMap[currentPixel] =
              DistMax * DistMax / (double) (size * ScaleX * ScaleY) * 0.785398;

        } else {
          // En 3D
          DistanceMap[currentPixel] =
              DistMax * DistMax * DistMax /
              (double) (size * ScaleX * ScaleY * ScaleZ) * 0.523599;
        }
        continue;
      }
      if (Allongement == 2) {
        // tortuosity
        int X2, Y2, Z2;
        X2 = IndStart % W;
        Y2 = (IndStart % (W * H) - X2) / W;
        Z2 = (IndStart - X2 - Y2 * W) / (W * H);
        double Eucl =
            std::sqrt((double) ((Xtort - X2) * (Xtort - X2) * ScaleX * ScaleX +
                                (Ytort - Y2) * (Ytort - Y2) * ScaleY * ScaleY +
                                (Ztort - Z2) * (Ztort - Z2) * ScaleZ * ScaleZ));
        if (Eucl != 0)
          DistanceMap[currentPixel] = (double) (DistMax / Eucl);
        else
          DistanceMap[currentPixel] = (double) (DistMax / 0.01);
        continue;
      }
      if (Allongement == 3) {
        // extremities
        DistanceMap[currentPixel] = 0;
        continue;
      }

    } while (!fifoSave->empty());

    // Extremities (first extremity is set to 1, the other one is set to 2. The
    // rest of the CC is set to 0)
    if (Allongement == 3) {
      int IndExt            = Xtort + Ytort * W + Ztort * W * H;
      DistanceMap[IndStart] = 1;
      DistanceMap[IndExt]   = 2;
      // JOE int X0,Y0,Z0;
      // X0=IndStart%W;
      // Y0=(IndStart%(W*H)-X0)/W;
      // Z0=(IndStart-X0-Y0*W)/(W*H);
    }

    return;
  } // END GeodesicPathCC

  //---------------------------------------------------------------------------------------------------------------------------------------------------------

  void GeodesicPathFlatZones(double *DistanceMap, double *LabelMap,
                             std::queue<int> *fifoSave, int W, int H, int D,
                             int BaryX, int BaryY, int BaryZ,
                             int Allongement = 0, float ScaleX = 1,
                             float ScaleY = 1, float ScaleZ = 1)
  {
    if (fifoSave->empty())
      return;

    int   IndStart, currentPixel, X, Y, Z, Xtort, Ytort, Ztort;
    float DistMax = -1, Dist;

    // Find the BaryCentre
    std::queue<int> fifoCurrent = *fifoSave; // Copy of fifoCurrent

    // Find the first Pixel (farest from the barycenter -- using Eucludean
    // distance because barycenter could not be inside the object)
    do {
      currentPixel = fifoCurrent.front();
      fifoCurrent.pop();
      X = currentPixel % W;
      Y = (currentPixel % (W * H) - X) / W;
      Z = (currentPixel - X - Y * W) / (W * H);
      // Dist = std::sqrt(pow((double)(X*ScaleX -
      // BaryX),2)+pow((double)(Y*ScaleY-BaryY),2)+pow((double)(Z*ScaleZ -
      // BaryZ),2));
      Dist = (float) std::sqrt(
          (double) ((X * ScaleX - BaryX) * (X * ScaleX - BaryX) +
                    (Y * ScaleY - BaryY) * (Y * ScaleY - BaryY) +
                    (Z * ScaleZ - BaryZ) * (Z * ScaleZ - BaryZ)));

      if (Dist > DistMax) {
        DistMax  = Dist;
        IndStart = currentPixel;
      }
    } while (!fifoCurrent.empty());

    // Get the max distance DistMax (geodesic diameter)
    bool NewDist;
    int  k, l, Ind, IndCurr, j;
    DistanceMap[IndStart] = 1;
    fifoCurrent.push(IndStart);
    DistMax = 1;

    do {
      currentPixel = fifoCurrent.front();
      fifoCurrent.pop();
      X = currentPixel % W;
      Y = (currentPixel % (W * H) - X) / W;
      Z = (currentPixel - X - Y * W) / (W * H);

      IndCurr = X + (Y * W) + (Z * W * H);

      Dist    = static_cast<float>(ImDtTypes<float>::max());
      NewDist = 0;

      // For all the neighbour
      for (j = -1; j <= 1; j++)
        if (Z + j >= 0 && Z + j < D)
          for (k = -1; k <= 1; k++)
            if (X + k >= 0 && X + k < W)
              for (l = -1; l <= 1; l++)
                if (Y + l >= 0 && Y + l < H && (k != 0 || l != 0 || j != 0)) {
                  Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;

                  if (LabelMap[Ind] == LabelMap[IndCurr]) {
                    if (DistanceMap[Ind] == -2) {
                      fifoCurrent.push(Ind);
                      DistanceMap[Ind] = -3;
                    } else if (DistanceMap[Ind] != 0 &&
                               DistanceMap[Ind] != -3) {
                      float D = std::sqrt((j * ScaleX * j * ScaleX) +
                                          (k * ScaleY) * (k * ScaleY) +
                                          (l * ScaleZ) * (l * ScaleZ));
                      if (NewDist == 0 || Dist > (DistanceMap[Ind] + D)) {
                        Dist    = (float) DistanceMap[Ind] + D;
                        NewDist = 1;
                      }
                    }

                    if (NewDist) {
                      // !(FLOAT_EQ_CORRIGE(Dist,DataTraits<float
                      // !>::default_value::max_value()))){ //==> Dist !=
                      // !max_value()
                      DistanceMap[X + Y * W + Z * W * H] = Dist;
                      if (Dist > DistMax) {
                        DistMax = Dist;
                        Xtort   = X;
                        Ytort   = Y;
                        Ztort   = Z;
                      }
                    }
                  }
                }
    } while (!fifoCurrent.empty());

    // Write on DistanceMap the Max Distance for all the pixel of the CC, we pop
    // fifoSave
    int size = (int) fifoSave->size();
    do {
      currentPixel = fifoSave->front();
      fifoSave->pop();

      // geodesic diameter
      if (Allongement == 0) {
        DistanceMap[currentPixel] = DistMax;
        continue;
      }
      if (Allongement == 1) {
        // Geodesic elongation //BMITOTO

        if (D == 1) {
          // 2D
          DistanceMap[currentPixel] =
              DistMax * DistMax / (double) (size * ScaleX * ScaleY) * 0.785398;
        } else {
          // 3D
          DistanceMap[currentPixel] =
              DistMax * DistMax * DistMax /
              (double) (size * ScaleX * ScaleY * ScaleZ) * 0.523599;
        }
        continue;
      }
      if (Allongement == 2) {
        // Tortuosity
        int X2, Y2, Z2;
        X2 = IndStart % W;
        Y2 = (IndStart % (W * H) - X2) / W;
        Z2 = (IndStart - X2 - Y2 * W) / (W * H);
        double Eucl =
            std::sqrt((double) ((Xtort - X2) * (Xtort - X2) * ScaleX * ScaleX +
                                (Ytort - Y2) * (Ytort - Y2) * ScaleY * ScaleY +
                                (Ztort - Z2) * (Ztort - Z2) * ScaleZ * ScaleZ));
        if (Eucl != 0)
          DistanceMap[currentPixel] = (double) (DistMax / Eucl);
        else
          DistanceMap[currentPixel] = (double) (DistMax / 0.01);
        continue;
      }
      if (Allongement == 3) {
        // Extremities (first extremity is set to 1, the other one is set to 2.
        // The rest of the CC is set to 0)
        DistanceMap[currentPixel] = 0;
        continue;
      }
    } while (!fifoSave->empty());

    // Extremities (first extremity is set to 1, the other one is set to 2. The
    // rest of the CC is set to 0)
    if (Allongement == 3) {
      int IndExt            = Xtort + Ytort * W + Ztort * W * H;
      DistanceMap[IndStart] = 1;
      DistanceMap[IndExt]   = 2;
      // JOE
      // int X0,Y0,Z0;
      // X0=IndStart%W;
      // Y0=(IndStart%(W*H)-X0)/W;
      // Z0=(IndStart-X0-Y0*W)/(W*H);
    }

    return;
  } // END GeodesicPathFlatZones

  //---------------------------------------------------------------------------------------------------------------------------------------------------------

  void t_ImGeodesicPathBinary(double *DistanceMap, int W, int H, int D,
                              int Allongement = 0, float ScaleX = 1,
                              float ScaleY = 1, float ScaleZ = 1)
  {
    std::queue<int> fifoCurrent, fifoSave;
    int             X, Y, Z, currentPixel, i, j, k, l, Ind;
    int             BaryX, BaryY, BaryZ;

    // For all pixels
    for (i = W * H * D - 1; i >= 0; i--)
      if (DistanceMap[i] == -1) {
        // We are on a CC
        fifoCurrent.push(i);
        fifoSave.push(i);

        DistanceMap[i] = -2;
        BaryX          = 0;
        BaryY          = 0;
        BaryZ          = 0;

        // We push all the CC
        do {
          currentPixel = fifoCurrent.front();
          fifoCurrent.pop();
          X = currentPixel % W;
          Y = (currentPixel % (W * H) - X) / W;
          Z = (currentPixel - X - Y * W) / (W * H);
          BaryX += X;
          BaryY += Y;
          BaryZ += Z;

          // For all the neigbour
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      if (DistanceMap[Ind] == -1) {
                        DistanceMap[Ind] = -2;
                        fifoCurrent.push(Ind);
                        fifoSave.push(Ind);
                      }
                    }
        } while (!fifoCurrent.empty());

        // One CC have been push, we compute it
        BaryX = (int) (BaryX * ScaleX / fifoSave.size());
        BaryY = (int) (BaryY * ScaleY / fifoSave.size());
        BaryZ = (int) (BaryZ * ScaleZ / fifoSave.size());
        GeodesicPathCC(DistanceMap, &fifoSave, W, H, D, BaryX, BaryY, BaryZ,
                       Allongement, ScaleX, ScaleY, ScaleZ);
      }
  }

  //---------------------------------------------------------------------------------------------------------------------------------------------------------

  void t_ImGeodesicPathOnLabels(double *DistanceMap, double *LabelMap, int W,
                                int H, int D, int Allongement = 0,
                                float ScaleX = 1, float ScaleY = 1,
                                float ScaleZ = 1)
  {
    std::queue<int> fifoCurrent, fifoSave;
    int             X, Y, Z, currentPixel, i, j, k, l, Ind, IndCurr;
    int             BaryX, BaryY, BaryZ;

    for (i = W * H * D - 1; i >= 0; i--) {
      // For all pixels
      if (DistanceMap[i] == -1) {
        // We are on a CC
        fifoCurrent.push(i);
        fifoSave.push(i);

        DistanceMap[i] = -2;
        BaryX          = 0;
        BaryY          = 0;
        BaryZ          = 0;

        // We push all the CC
        do {
          currentPixel = fifoCurrent.front();
          fifoCurrent.pop();
          X = currentPixel % W;
          Y = (currentPixel % (W * H) - X) / W;
          Z = (currentPixel - X - Y * W) / (W * H);
          BaryX += X;
          BaryY += Y;
          BaryZ += Z;

          IndCurr = (X) + (Y * W) + (Z * W * H);

          // For all the neigbour (8-connectivity in 2D or 26-connectivity in
          // 3D)
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      // Are pixels in the same flat zone?
                      if (DistanceMap[Ind] == -1 &&
                          LabelMap[Ind] == LabelMap[IndCurr]) {
                        DistanceMap[Ind] = -2;
                        fifoCurrent.push(Ind);
                        fifoSave.push(Ind);
                      }
                    }
        } while (!fifoCurrent.empty());

        // One CC have been push, we compute it
        BaryX = (int) (BaryX * ScaleX / fifoSave.size());
        BaryY = (int) (BaryY * ScaleY / fifoSave.size());
        BaryZ = (int) (BaryZ * ScaleZ / fifoSave.size());

        GeodesicPathFlatZones(DistanceMap, LabelMap, &fifoSave, W, H, D, BaryX,
                              BaryY, BaryZ, Allongement, ScaleX, ScaleY,
                              ScaleZ);
      }
    }
  }

  //---------------------------------------------------------------------------------------------------------------------------------------------------------

  template <class T>
  RES_T ImGeodesicPathOpening(const Image<UINT8> &imIn, double Lenght,
                              int Method, Image<T> &imOut, float ScaleX,
                              float ScaleY, float ScaleZ)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    // On passe par des buffers pour des questions de performance, En 3D, on
    // passe de 80 a 50 secondes
    typename Image<UINT8>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T>::lineType     pixelsOut = imOut.getPixels();

    // Calcul des niveaux de gris a reellement faire: si un niveau de gris n'est
    // pas present dans l'image initial, aucun changement dans les CC. Donc il
    // n'y a pas besoin de perdre du temps pour ne rien changer...
    UINT8            Hist[256] = {0};
    std::vector<int> levelToDo;

    for (size_t i = 0; i < imIn.getPixelCount(); ++i) {
      Hist[pixelsIn[i]] = 1;
    }

    for (int i = 0; i < 256; ++i)
      if (Hist[i] != 0)
        levelToDo.push_back(i);

    double *DistanceMap = new double[W * H * Z];
    if (DistanceMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    // For all the level which are needed
    for (int k = 0; k < (int) levelToDo.size(); k++) {
      // UseLess to compute the first level
      if (k == 0) {
        for (int i = W * H * Z - 1; i >= 0; i--)
          pixelsOut[i] = levelToDo[k]; // ImOut = first level
        continue;
      }

      for (int i = W * H * Z - 1; i >= 0; i--)
        // Threshold
        DistanceMap[i] = ((pixelsIn[i] >= levelToDo[k]) ? (-1) : (0));
      t_ImGeodesicPathBinary(DistanceMap, W, H, Z, Method, ScaleX, ScaleY,
                             ScaleZ);
      for (int i = W * H * Z - 1; i >= 0; i--)
        if (DistanceMap[i] >= Lenght)
          pixelsOut[i] = levelToDo[k];
    }
    delete[] DistanceMap;
    return RES_OK;
  }

  template <class T>
  RES_T ImUltimateGeodesicPathOpening(const Image<UINT8> &imIn,
                                      Image<UINT8> &imTrans, Image<T> &imInd,
                                      float ScaleX, float ScaleY, float ScaleZ,
                                      int stop, int lambdaAttribute,
                                      int takeMin)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imTrans)
    ASSERT_SAME_SIZE(&imIn, &imInd)

    ImageFreezer freeze(imTrans);
    ImageFreezer freeze_2(imInd);
    fill(imTrans, UINT8(0));
    fill(imInd, T(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    typename Image<UINT8>::lineType pixelsIn    = imIn.getPixels();
    typename Image<UINT8>::lineType pixelsTrans = imTrans.getPixels();
    typename Image<T>::lineType     pixelsInd   = imInd.getPixels();

    // Calcul des niveaux de gris a reellement faire: si un niveau de gris n'est
    // pas present dans l'image initial, aucun changement dans les CC. Donc il
    // n'y a pas besoin de perdre du temps pour ne rien changer...
    UINT8            Hist[256] = {0};
    std::vector<int> levelToDo;
    // Il faut obligatoirement faire le level 0 pour des questions
    // d'initialisation
    levelToDo.push_back(0);

    for (size_t i = 0; i < imIn.getPixelCount(); ++i) {
      Hist[pixelsIn[i]] = 1;
    }

    for (int i = 0; i < 256; i++)
      if (Hist[i] != 0)
        levelToDo.push_back(i + 1);

    double *DistanceMap = new double[W * H * Z];
    if (DistanceMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    std::vector<std::vector<MyPOINTS>> PathOp(W * H * Z);
    MyPOINTS                           Pt;

    // For all the level which are needed
    for (int k = 0; k < (int) levelToDo.size(); k++) {
      for (size_t i = 0; i < imIn.getPixelCount(); ++i)
        DistanceMap[i] =
            ((pixelsIn[i] >= levelToDo[k]) ? (-1) : (0)); // Threshold

      // sinon aucune propogation car l'image est noir-->pour ajouter le dernier
      // maillon Dist=0
      if (k != (int) levelToDo.size() - 1)
        t_ImGeodesicPathBinary(DistanceMap, W, H, Z, 0, ScaleX, ScaleY, ScaleZ);

      for (int i = W * H * Z - 1; i >= 0; i--) {
        if (levelToDo[k] != 0)
          Pt = PathOp[i].back();
        if (levelToDo[k] == 0 || !FLOAT_EQ_CORRIGE(Pt.Dist, DistanceMap[i])) {
          Pt.Dist  = (int) DistanceMap[i];
          Pt.Seuil = levelToDo[k];
          PathOp[i].push_back(Pt);
        }
      }
    }
    delete[] DistanceMap;

    //****************************************************
    // Computation of the Ultimate Path
    //****************************************************
    // UO initialisation
    Image<UINT8> imPOold(imIn);
    copy(imIn, imPOold);
    typename Image<UINT8>::lineType pixelsPOold = imPOold.getPixels();

    int   ValPO, Sub;
    INT8 *Accumulation = new INT8[W * H * Z];
    if (Accumulation == NULL) {
      // MORPHEE_REGISTER_ERROR("Accumulation allocation");
      return RES_ERR_BAD_ALLOCATION;
    }
    // init arrray
    std::memset(Accumulation, 0, W * H * Z);

    // if the Stop criteria is not defined we set it to max(W,H) -1
    if (stop < 0)
      stop = std::max(W, std::max(H, Z)) - 1;
    if (stop > std::max(W, std::max(H, Z)) - 1)
      stop = std::max(W, std::max(H, Z)) - 1;

    for (int Lenght = 1; Lenght < stop; Lenght++) {
      for (size_t i = 0; i < imIn.getPixelCount(); ++i) {
        // Calcul de la valeur du PathOp pour la distance Lenght pour le pixel i
        // A la premiere valeur qui ne respecte pas le critere, on s'arrete
        if (takeMin == 1)
          for (int k = 0;; k++) {
            if (PathOp[i][k].Dist < Lenght) {
              ValPO = PathOp[i][k].Seuil - 1;
              break;
            }
          }
        else {
          for (int k = PathOp[i].size() - 2; k >= 0; k--)
            if (PathOp[i][k].Dist >= Lenght) {
              ValPO = PathOp[i][k + 1].Seuil - 1;
              break;
            }
        }
        if (ValPO < 0)
          ValPO = 0;

        // Initialisation des images a 0
        if (Lenght == 1) {
          pixelsInd[i]   = 0;
          pixelsTrans[i] = 0;
        }

        // On fait la soustraction entre le PO lambda-1 et le PO Lambda
        Sub = pixelsPOold[i] - ValPO;
        // On ecrit si on a un residu plus grand que l'ancien  (max)
        if (pixelsTrans[i] <= Sub && Sub > 0 && Accumulation[i] <= 0) {
          pixelsTrans[i]  = Sub;
          pixelsInd[i]    = Lenght + 1;
          Accumulation[i] = lambdaAttribute;
        } else {
          if (Accumulation[i] >= 1) {
            // Ou si l'accumulation est active
            // On accumule le contraste
            pixelsTrans[i] = pixelsTrans[i] + Sub;
            // On ecrit l'indicatrice
            pixelsInd[i] = Lenght + 1;
          }
        }
        if (Sub == 0)
          if (Accumulation[i] > -1)
            Accumulation[i]--;

        // On copy la nouvelle valeur de l'ouverture ultime
        pixelsPOold[i] = ValPO;
      }
    }

    delete[] Accumulation;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImGeodesicElongation(const Image<T1> &imIn, Image<T2> &imOut,
                             int sliceBySlice, double dz_over_dx) // BMI SK
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    // On passe par des buffers pour des questions de performance, En 3D, on
    // passe de 80 a 50 secondes
    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    double *DistanceMap = new double[W * H * Z];
    if (DistanceMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    if (!sliceBySlice) {
      // 3D
      for (size_t i = 0; i < imIn.getPixelCount(); ++i)
        // Threshold
        DistanceMap[i] = ((pixelsIn[i] >= 128) ? (-1) : (0));
      t_ImGeodesicPathBinary(DistanceMap, W, H, Z, 1, 1, 1, (float) dz_over_dx);
      for (size_t i = 0; i < imOut.getPixelCount(); ++i) {
        if (DistanceMap[i] >= ImDtTypes<T2>::max())
          pixelsOut[i] = ImDtTypes<T2>::max();
        else
          pixelsOut[i] = (T2) DistanceMap[i];
      }
    } else {
      // 2D slice by slice
      int slice, Ind, j, i;
      for (slice = 0; slice < Z; slice++) {
        Ind = slice * W * H;
        for (j = 0, i = Ind; j < W * H; ++j, ++i)
          DistanceMap[j] = ((pixelsIn[i] >= 128) ? (-1) : (0));
        t_ImGeodesicPathBinary(DistanceMap, W, H, 1, 1);

        for (j = 0, i = Ind; i < Ind + W * H; ++i, ++j)
          pixelsOut[i] = (T2) DistanceMap[j];
      }
    }

    delete[] DistanceMap;
    return RES_OK;
  }

  // BMI BEGIN ImGeodesicExtremities
  template <class T1, class T2>
  RES_T ImGeodesicExtremities(const Image<T1> &imIn, Image<T2> &imOut,
                              int sliceBySlice, double dz_over_dx)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    // On passe par des buffers pour des questions de performance, En 3D, on
    // passe de 80 a 50 secondes
    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    double *DistanceMap = new double[W * H * Z];
    if (DistanceMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    if (!sliceBySlice) {
      // 3D
      for (size_t i = 0; i < imIn.getPixelCount(); ++i)
        // Threshold
        DistanceMap[i] = ((pixelsIn[i] >= 128) ? (-1) : (0));
      t_ImGeodesicPathBinary(DistanceMap, W, H, Z, 3, 1, 1, (float) dz_over_dx);
      for (size_t i = 0; i < imOut.getPixelCount(); ++i) {
        if (DistanceMap[i] >= ImDtTypes<T2>::max())
          pixelsOut[i] = ImDtTypes<T2>::max();
        else
          pixelsOut[i] = (T2) DistanceMap[i];
      }
    } else {
      // 2D slice by slice
      int slice, Ind, j, i;
      for (slice = 0; slice < Z; slice++) {
        Ind = slice * W * H;
        for (j = 0, i = Ind; j < W * H; ++j, ++i)
          DistanceMap[j] = ((pixelsIn[i] >= 128) ? (-1) : (0));
        t_ImGeodesicPathBinary(DistanceMap, W, H, 1, 3);

        for (j = 0, i = Ind; i < Ind + W * H; ++i, ++j)
          pixelsOut[i] = (T2) DistanceMap[j];
      }
    }

    delete[] DistanceMap;
    return RES_OK;
  }

  //---------------------------------------------------------------------------------------------------------------------------------------------------------

  // BMI: ImLabelFlatZonesWithElongation calls ImGeodesicPathOnLabels that calls
  // GeodesicPathFlatZones
  // BMI: ImGeodesicPathOpening calls ImGeodesicPathBinary that calls
  // GeodesicPathCC t_ImGeodesicPathOnLabels : used to compute Elongation.
  // Modification in order to get diameter
  // BMI: ImLabelFlatZonesWithElongation calls ImGeodesicDiameterOnLabels that
  // calls GeodesicPathFlatZones

  template <class T1, class T2>
  RES_T ImLabelFlatZonesWithElongation(const Image<T1> &imIn, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    // On passe par des buffers pour des questions de performance, En 3D, on
    // passe de 80 a 50 secondes
    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    double *DistanceMap = new double[W * H * Z];
    double *LabelMap    = new double[W * H * Z];

    if (DistanceMap == NULL || LabelMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap and LabelMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    int slice, Ind, j, i;

    for (slice = 0; slice < Z; slice++) {
      Ind = slice * W * H;
      for (j = 0, i = Ind; j < W * H; ++j, ++i) {
        // Process all pixels
        DistanceMap[j] = ((pixelsIn[i] >= 0) ? (-1) : (0));
        // Pixel labels
        LabelMap[j] = (pixelsIn[i]);
      }
      // Depth = 1 (image2D), Alongement=1 (geodesic elongation)
      t_ImGeodesicPathOnLabels(DistanceMap, LabelMap, W, H, 1, 1);

      for (j = 0, i = Ind; i < Ind + W * H; ++i, ++j) {
        pixelsOut[i] = (T2) DistanceMap[j];
      }
    }

    delete[] DistanceMap;
    delete[] LabelMap;

    return RES_OK;
  }

  //---------------------------------------------------------------------------------------------------------------------------------------------------------
  // ..................................................
  // BMI END EXTREMITIES
  // ..................................................
  // Alongement = 0 (diametre geodesique), 1 (elongation), 2 (tortuosite), 3
  // (extremites 1 y 2 du chemin ayant servi a calculer le diametre geodesique)
  // BMI: ImLabelFlatZonesWithExtremities calls ImGeodesicPathOnLabels that
  // calls GeodesicPathFlatZones(Alongement = 3) BMI:
  // ImLabelFlatZonesWithElongation calls ImGeodesicPathOnLabels that calls
  // GeodesicPathFlatZones(Alongement = 1)
  // BMI: ImGeodesicPathOpening calls ImGeodesicPathBinary that calls
  // GeodesicPathCC t_ImGeodesicPathOnLabels : used to compute Elongation.
  // Modification in order to get diameter

  template <class T1, class T2>
  RES_T ImLabelFlatZonesWithExtremities(const Image<T1> &imIn, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    // On passe par des buffers pour des questions de performance, En 3D, on
    // passe de 80 a 50 secondes
    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    double *DistanceMap = new double[W * H * Z];
    double *LabelMap    = new double[W * H * Z];

    if (DistanceMap == NULL || LabelMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap and LabelMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    int slice, Ind, j, i;

    for (slice = 0; slice < Z; slice++) {
      Ind = slice * W * H;
      for (j = 0, i = Ind; j < W * H; ++j, ++i) {
        // Process all pixels
        DistanceMap[j] = ((pixelsIn[i] >= 0) ? (-1) : (0));
        // Pixel labels
        LabelMap[j] = (pixelsIn[i]);
      }
      // le 1 est pour une image 2D, le 3 est pour les extremites
      t_ImGeodesicPathOnLabels(DistanceMap, LabelMap, W, H, 1, 3);

      for (j = 0, i = Ind; i < Ind + W * H; ++i, ++j) {
        pixelsOut[i] = (T2) DistanceMap[j];
      }
    }

    delete[] DistanceMap;
    delete[] LabelMap;

    return RES_OK;
  }

  // ..................................................
  // BMI END EXTREMITIES
  // ..................................................
  template <class T1, class T2>
  RES_T ImLabelFlatZonesWithGeodesicDiameter(const Image<T1> &imIn,
                                             Image<T2> &      imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    // On passe par des buffers pour des questions de performance, En 3D, on
    // passe de 80 a 50 secondes
    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    double *DistanceMap = new double[W * H * Z];
    double *LabelMap    = new double[W * H * Z];

    if (DistanceMap == NULL || LabelMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap and LabelMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    int slice, Ind, j, i;

    for (slice = 0; slice < Z; slice++) {
      Ind = slice * W * H;
      for (j = 0, i = Ind; j < W * H; ++j, ++i) {
        // Process all pixels
        DistanceMap[j] = ((pixelsIn[i] >= 0) ? (-1) : (0));
        // Pixel labels
        LabelMap[j] = pixelsIn[i];
      }

      // le 1 est pour une image 2D, le 0 est pour le diametre geodesique
      t_ImGeodesicPathOnLabels(DistanceMap, LabelMap, W, H, 1, 0);

      for (j = 0, i = Ind; i < Ind + W * H; ++i, ++j) {
        pixelsOut[i] = (T2) DistanceMap[j];
      }
    }

    delete[] DistanceMap;
    delete[] LabelMap;

    return RES_OK;
  }

  //---------------------------------------------------------------------------------------------------------------------------------------------------------

  template <class T>
  RES_T ImGeodesicTortuosity(const Image<UINT8> &imIn, Image<T> &imOut,
                             int sliceBySlice = 0)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    // On passe par des buffers pour des questions de performance, En 3D, on
    // passe de 80 a 50 secondes
    typename Image<UINT8>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T>::lineType     pixelsOut = imOut.getPixels();

    double *DistanceMap = new double[W * H * Z];
    if (DistanceMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    if (!sliceBySlice) {
      for (size_t i = 0; i < imIn.getPixelCount(); i++)
        // Threshold
        DistanceMap[i] = ((pixelsIn[i] >= 128) ? (-1) : (0));
      t_ImGeodesicPathBinary(DistanceMap, W, H, Z, 2, 1, 1, 1);
      for (size_t i = 0; i < imOut.getPixelCount(); i++)
        pixelsOut[i] = (T)(DistanceMap[i]);
    } else {
      int slice, Ind, j, i;
      for (slice = 0; slice < Z; slice++) {
        Ind = slice * W * H;
        for (j = 0, i = Ind; j < W * H; ++j, ++i)
          DistanceMap[j] = ((pixelsIn[i] >= 128) ? (-1) : (0));
        t_ImGeodesicPathBinary(DistanceMap, W, H, 1, 2);

        for (j = 0, i = Ind; i < Ind + W * H; ++i, ++j)
          pixelsOut[i] = (T)(DistanceMap[j]);
      }
    }

    delete[] DistanceMap;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImGeodesicDiameter(const Image<T1> &imIn, Image<T2> &imOut,
                           int sliceBySlice, double dz_over_dx)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    // On passe par des buffers pour des questions de performance, En 3D, on
    // passe de 80 a 50 secondes
    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    double *DistanceMap = new double[W * H * Z];
    if (DistanceMap == NULL) {
      // MORPHEE_REGISTER_ERROR("DistanceMap allocation");
      return RES_ERR_BAD_ALLOCATION;
    }

    if (!sliceBySlice) {
      for (size_t i = 0; i < imIn.getPixelCount(); i++)
        // Threshold
        DistanceMap[i] = ((pixelsIn[i] >= 128) ? (-1) : (0));
      t_ImGeodesicPathBinary(DistanceMap, W, H, Z, 0, 1, 1, (float) dz_over_dx);
      for (size_t i = 0; i < imOut.getPixelCount(); i++)
        pixelsOut[i] = (T2) DistanceMap[i];
    } else {
      int slice, Ind, j, i;
      for (slice = 0; slice < Z; slice++) {
        Ind = slice * W * H;
        for (j = 0, i = Ind; j < W * H; ++j, ++i)
          DistanceMap[j] = ((pixelsIn[i] >= 128) ? (-1) : (0));
        t_ImGeodesicPathBinary(DistanceMap, W, H, 1, 0);

        for (j = 0, i = Ind; i < Ind + W * H; ++i, ++j)
          pixelsOut[i] = (T2) DistanceMap[j];
      }
    }

    delete[] DistanceMap;
    return RES_OK;
  }

} // namespace smil

/*
template<class T1, class T2, class T3>
    RES_C  t_ImUltimateGeodesicPathOpeningFAST(const Image<T1> & imIn,Image<T2>
& imTrans,Image<T3> & imInd, int stop, int lambdaAttribute,int takeMin )
  {
    MORPHEE_ENTER_FUNCTION("t_ImGeodesicPathUltimateOpening(template)");

    //Check inputs
    if( ! imIn.isAllocated() || ! imTrans.isAllocated()|| ! imInd.isAllocated()
){ MORPHEE_REGISTER_ERROR("Image not allocated"); return RES_NOT_ALLOCATED;
    }
    if(!t_CheckWindowSizes(imIn, imTrans) || !t_CheckWindowSizes(imIn, imInd)){
      MORPHEE_REGISTER_ERROR("Bad window sizes");
      return RES_ERROR_BAD_WINDOW_SIZE;
    }

    int W = imIn.getWxSize();
    int H = imIn.getWySize();
    int Z = imIn.getWzSize();
    int i,k;
    typename Image<T1>::const_iterator itIn,itEnd;
    typename Image<T2>::iterator itTrans;
    typename Image<T3>::iterator itInd;

    //Calcul des niveaux de gris a reellement faire: si un niveau de gris n'est
pas
    //present dans l'image initial, aucun changement dans les CC. Donc il n'y a
pas
    //besoin de perdre du temps pour ne rien changer...
    UINT8 Hist[256];
    std::vector<int>levelToDo;
    levelToDo.push_back(0);   //Il faut obligatoirement faire le level 0 pour
des questions d'initialisation for(i=0;i<256;i++) Hist[i]=0;    //Initialement
NDG non present for(itIn=imIn.begin(),itEnd=imIn.end(); itIn!=itEnd; ++itIn)
      Hist[*itIn]=1;
    for(i=0;i<256;i++)
      if(Hist[i]!=0)
        levelToDo.push_back(i+1);

    double *DistanceMap = new double[W*H*Z];
    if(DistanceMap==NULL){
      MORPHEE_REGISTER_ERROR("DistanceMap allocation");
      return RES_ERROR_MEMORY;
    }

    std::vector<std::vector<MyPOINTS> > PathOp(W*H*Z);
    MyPOINTS Pt;


    //For all the level which are needed
    for(k = 0 ; k<(int)levelToDo.size() ; k++){
      for(i=0,itIn=imIn.begin(),itEnd=imIn.end();itIn!=itEnd;++itIn,++i)
        DistanceMap[i]=((*itIn>=levelToDo[k])?(-1):(0));    //Threshold

      if(k != (int)levelToDo.size()-1)  // sinon aucune propogation car l'image
est noir-->pour ajouter le dernier maillon Dist=0
        t_ImGeodesicPathBinary(DistanceMap,W,H,Z);

      for(i=W*H*Z-1;i>=0;i--){
        if(levelToDo[k]!=0)
          Pt=PathOp[i].back();
        if(levelToDo[k]==0 || !FLOAT_EQ_CORRIGE2 (Pt.Dist ,
DistanceMap[i],0.5)){ Pt.Dist=DistanceMap[i]; Pt.Seuil=levelToDo[k];
          PathOp[i].push_back(Pt);
        }
      }
    }
    delete []DistanceMap;

    // ****************************************************
    // Computation of the Ultimate Path
    // ****************************************************
    //UO initialisation
    UINT8 *imPOold = new UINT8[W*H*Z];
    if(imPOold==NULL){
      MORPHEE_REGISTER_ERROR("imPOold allocation");
      return RES_ERROR_MEMORY;
    }

    int ValPO,Sub;
    INT8 *Accumulation=new INT8[W*H*Z];
    if(Accumulation==NULL){
      MORPHEE_REGISTER_ERROR("Accumulation allocation");
      delete []imPOold;
      return RES_ERROR_MEMORY;
    }

    for(i=0,itIn=imIn.begin(),itEnd=imIn.end(); itIn!=itEnd; ++itIn,++i){
      imPOold[i]=*itIn;   //Copy the image
      Accumulation[i]=0;  //No accumulation
    }

    //if the Stop criteria is not defined we set it to max(W,H) -1
    if(stop <0) stop = std::max(W,std::max(H,Z))-1;
    if(stop > std::max(W,std::max(H,Z))-1) stop = std::max(W,std::max(H,Z))-1;

    int D,MaxContrast,Contrast;
    for(i=0,itInd=imInd.begin(),itTrans=imTrans.begin(),itIn=imIn.begin(),itEnd=imIn.end();
itIn!=itEnd; ++itIn,++i,++itInd,++itTrans){ MaxContrast = -1;
      for(k=PathOp[i].size()-2;k>=0;k--){
        Contrast = PathOp[i][k+1].Seuil - PathOp[i][k].Seuil;
        if(Contrast >= MaxContrast && PathOp[i][k].Dist < stop){
          MaxContrast = Contrast;
          D = PathOp[i][k].Dist;
        }
      }
      if(MaxContrast <0){
        *itInd = 0;
        *itTrans = 0;
      }
      else{
        *itInd = D;
        *itTrans = MaxContrast;
      }
    }

    delete []Accumulation;
    return RES_OK;
  }
*/

#endif
