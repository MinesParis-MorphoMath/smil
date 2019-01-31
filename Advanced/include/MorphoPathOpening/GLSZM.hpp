#ifndef __PO_GLSZM_T_HPP__
#define __PO_GLSZM_T_HPP__

#include "Core/include/DCore.h"

namespace smil
{
#define round(x) ((int) ((x) + 0.5))
  double Sum = 1;

  // -------------------------------------------- Calcul des caracteristiques
  // -------------------------------------------- */ F0 - Small Zone Emphasis,
  // SZE. Petites zones.
  //@return SZE.*/
  double SZE(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += matrix[s + n * Width] / (double) pow((double) (s + 1), 2.0);
    return val / Sum;
  }

  // F1 - Large Zone Emphasis, LZE. Grande zone.
  // @return LZE.*/
  double LZE(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += matrix[s + n * Width] * (double) pow((double) (s + 1), 2.0);
    return val / Sum;
  }

  /** F2 - Low Gray level Zone Emphasis, LGZE.
   * @return LGZE.*/
  double LGZE(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += matrix[s + n * Width] / (double) pow((double) (n + 1), 2.0);
    return val / Sum;
  }

  /** F3 - High Gray level Zone Emphasis, HGZE.
   * @return HGZE.*/
  double HGZE(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += matrix[s + n * Width] * (double) pow((double) (n + 1), 2.0);
    return val / Sum;
  }

  /** F4 - Small Zone Low Gray level Emphasis, SZLGE.
   * @return SZLGE.*/
  double SZLGE(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += matrix[s + n * Width] / (double) (pow((double) (n + 1), 2.0) *
                                                 pow((double) (s + 1), 2.0));
    return val / Sum;
  }

  /** F5 - Small Zone High Gray level Emphasis, SZHGE.
   * @return SZHGE.*/
  double SZHGE(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += matrix[s + n * Width] *
               (double) pow((double) (n + 1) / (double) (s + 1), 2.0);
    return val / Sum;
  }

  /** F6 - Large Zone Low Gray level Emphasis, LZLGE.
   * @return LSLGLE.*/
  double LZLGE(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += matrix[s + n * Width] *
               (double) pow((double) (s + 1) / (double) (n + 1), 2.0);
    return val / Sum;
  }

  /** F7 - Large Zone High Gray level Emphasis, LZHGE.
   * @return LZHGE.*/
  double LZHGE(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += matrix[s + n * Width] * (double) pow((double) (n + 1), 2.0) *
               pow((double) (s + 1), 2.0);
    return val / Sum;
  }

  /** F8 - Gray Level Non Uniform, GLNU. Homogeneite spectrale.
   * @return GLNU.*/
  double GLNU(int *matrix, int Width, int nbGrayLevel)
  {
    double v, val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++) {
      v = 0.0;
      for (int s = 0; s < Width; s++)
        v += matrix[s + n * Width];
      val += v * v;
    }
    return val / Sum;
  }

  /** F9 - Size Zone Non Uniform, SZNU. Uniformite.
   * @return SZNU.*/
  double SZNU(int *matrix, int Width, int nbGrayLevel)
  {
    double v, val = 0.0;
    for (int s = 0; s < Width; s++) {
      v = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        v += matrix[s + n * Width];
      val += v * v;
    }
    return val / Sum;
  }

  /** F10 - Zone Percentage, ZPC. Egalite des isotailles (pourcentage
   * primitives).
   * @return ZPC.*/
  double ZPC(int *matrix, int Width, int nbGrayLevel)
  {
    double val = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        val += (double) (s + 1) * matrix[s + n * Width];
    return Sum / val;
  }

  /** F11 - Methode qui calcule le barycentre sur les niveaux de gris.
   * @return Le barycentre sur les niveaux de gris.*/
  double BARYGL(int *matrix, int Width, int nbGrayLevel)
  {
    double mean = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        mean += (double) (n + 1) * matrix[s + n * Width];
    return mean / Sum;
  }

  /** F12 - Methode qui calcule le barycentre sur les tailles.
   * @return Le barycentre sur les tailles.*/
  double BARYS(int *matrix, int Width, int nbGrayLevel)
  {
    double mean = 0.0;
    for (int n = 0; n < nbGrayLevel; n++)
      for (int s = 0; s < Width; s++)
        mean += (double) (s + 1) * matrix[s + n * Width];
    return mean / Sum;
  }

  void ComputeFeatures(int *Matrix, int W, int H, char *szFileName)
  {
    double Features[12];

    Features[0]  = SZE(Matrix, W, H);
    Features[1]  = LZE(Matrix, W, H);
    Features[2]  = LGZE(Matrix, W, H);
    Features[3]  = HGZE(Matrix, W, H);
    Features[4]  = SZLGE(Matrix, W, H);
    Features[5]  = SZHGE(Matrix, W, H);
    Features[6]  = LZLGE(Matrix, W, H);
    Features[7]  = LZHGE(Matrix, W, H);
    Features[8]  = GLNU(Matrix, W, H);
    Features[9]  = SZNU(Matrix, W, H);
    Features[10] = BARYGL(Matrix, W, H);
    Features[11] = BARYS(Matrix, W, H);
    // Features[12] = VARGL(int *Matrix, int W,int H);
    // Features[13] = VARS(int *Matrix, int W,int H);

    FILE *fic = fopen(szFileName, "a");
    for (int i = 0; i < 12; i++)
      fprintf(fic, "%f\t", Features[i]);
    fprintf(fic, "%f\n", Features[11]);
    fclose(fic);
  }

  int ComputeBinaryCC(int *DistanceMap, std::queue<int> *fifoSave, int W, int H,
                      int D, int BaryX, int BaryY, int BaryZ,
                      int GeodesicMethod = 0)
  {
    if (fifoSave->empty())
      return 0;

    int IndStart, currentPixel, X, Y, Z, Xtort, Ytort, Ztort;
    float DistMax = -1, Dist;

    // Find the BaryCentre
    std::queue<int> fifoCurrent = *fifoSave; // Copy of fifoCurrent

    // Find the first Pixel (farest)
    do {
      currentPixel = fifoCurrent.front();
      fifoCurrent.pop();
      X = currentPixel % W;
      Y = (currentPixel % (W * H) - X) / W;
      Z = (currentPixel - X - Y * W) / (W * H);
      // Dist = sqrt(pow((double)(X*ScaleX -
      // BaryX),2)+pow((double)(Y*ScaleY-BaryY),2)+pow((double)(Z*ScaleZ -
      // BaryZ),2));
      Dist = (float) sqrt((double) ((X - BaryX) * (X - BaryX) +
                                    (Y - BaryY) * (Y - BaryY) +
                                    (Z - BaryZ) * (Z - BaryZ)));

      if (Dist > DistMax) {
        DistMax  = Dist;
        IndStart = currentPixel;
      }
    } while (!fifoCurrent.empty());

    // Get the max distance
    bool NewDist;
    int k, l, Ind, j;
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
                  if (DistanceMap[Ind] == -1) {
                    fifoCurrent.push(Ind);
                    DistanceMap[Ind] = -2;
                  } else if (DistanceMap[Ind] != 0 && DistanceMap[Ind] != -2) {
                    float D = sqrt((float) ((j * j) + (k) * (k) + (l) * (l)));
                    if (NewDist == 0 || Dist > (DistanceMap[Ind] + D)) {
                      Dist    = (float) DistanceMap[Ind] + D;
                      NewDist = 1;
                    }
                  }

                  // !(FLOAT_EQ_CORRIGE(Dist,DataTraits<F_SIMPLE
                  // !>::default_value::max_value()))){ //==> Dist != max_value()
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
    int res;
    do {
      currentPixel = fifoSave->front();
      fifoSave->pop();
      if (GeodesicMethod == 1) {
        res = (int) DistMax;
      } else if (GeodesicMethod == 2) {
        // En 2D
        if (D == 1)
          res = (int) (DistMax * DistMax / (double) (size) *0.785398);
        // En 3D
        else
          res = (int) (DistMax * DistMax * DistMax / (double) (size) *0.523599);

      } else {
        int X2, Y2, Z2;
        X2          = IndStart % W;
        Y2          = (IndStart % (W * H) - X2) / W;
        Z2          = (IndStart - X2 - Y2 * W) / (W * H);
        double Eucl = sqrt((double) ((Xtort - X2) * (Xtort - X2) +
                                     (Ytort - Y2) * (Ytort - Y2) +
                                     (Ztort - Z2) * (Ztort - Z2)));
        if (Eucl != 0)
          res = (DistMax / Eucl) * 100;
        else
          res = (DistMax / 0.01) * 100;
      }
    } while (!fifoSave->empty());

    return res;
  }

  double ComputeBinary(int *pixelDone, int W, int H, int D, int *Matrix, int mW,
                       SMIL_UNUSED int mH, int Level, int GeodesicMethod = 0)
  {
    std::queue<int> fifoCurrent, fifoSave;
    int X, Y, Z, currentPixel, i, j, k, l, Ind;
    int BaryX, BaryY, BaryZ, NbPixel, Max = 0;

    for (i = W * H * D - 1; i >= 0; i--) { // For all pixels
      if (pixelDone[i] == 1) {
        // We are on a CC
        fifoCurrent.push(i);
        if (GeodesicMethod)
          fifoSave.push(i);

        pixelDone[i] = -1;
        BaryX        = 0;
        BaryY        = 0;
        BaryZ        = 0;
        NbPixel      = 0;

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
          NbPixel++;

          // For all the neigbour
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      if (pixelDone[Ind] == 1) {
                        pixelDone[Ind] = -1;
                        fifoCurrent.push(Ind);
                        if (GeodesicMethod)
                          fifoSave.push(Ind);
                      }
                    }
        } while (!fifoCurrent.empty());

        if (GeodesicMethod) {
          // One CC have been push, we compute it
          BaryX   = (int) (BaryX / fifoSave.size());
          BaryY   = (int) (BaryY / fifoSave.size());
          BaryZ   = (int) (BaryZ / fifoSave.size());
          NbPixel = ComputeBinaryCC(pixelDone, &fifoSave, W, H, D, BaryX, BaryY,
                                    BaryZ, GeodesicMethod);
        }

        if (NbPixel > Max)
          Max = NbPixel;

        // On accumule
        if (Matrix != 0)
          Matrix[NbPixel + Level * mW]++;
      }
    }
    return Max;
  }

  double Compute2D(UINT8 *bufferIn, int W, int H, int Z, int NbNDG, int *Matrix,
                   int mW, int mH, int GeodesicMethod = 0)
  {
    int *pixelDone = new int[W * H];
    int i, k, j, Level;
    double Max, M;

    Max = 0;

    for (Level = 1; Level <= NbNDG; Level++) {
      // for all slide
      for (k = 0; k < Z; k++) {
        int NbPix = 0;
        for (j = 0; j < H; j++) {
          for (i = 0; i < W; i++) {
            if (bufferIn[i + j * W + k * W * H] >= Level) {
              pixelDone[i + j * W] = 1;
              NbPix++;
            } else
              pixelDone[i + j * W] = 0;
          }
        }
        M = ComputeBinary(pixelDone, W, H, 1, Matrix, mW, mH, Level - 1,
                          GeodesicMethod);
        if (M > Max)
          Max = M;
      }
    }

    delete[] pixelDone;
    return Max;
  }

  template <class T1>
  RES_T Thibault_GLSZM(const Image<T1> &imIn, int NbNDG, char *szFileName)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();
    int i, DMax;

    typename Image<T1>::lineType pixelsIn = imIn.getPixels();
    UINT8 *bufferIn                       = new UINT8[W * H * Z];

    // Sous echantilonnage
    for (i = W * H * Z - 1; i >= 0; i--)
      bufferIn[i] = (int) round((pixelsIn[i] / 255.0) * NbNDG);

    DMax = Compute2D(bufferIn, W, H, Z, NbNDG, 0, 0, 0, false);

    int *Matrix = new int[DMax * NbNDG];

    for (i = NbNDG * DMax - 1; i >= 0; i--)
      Matrix[i] = 0;

    Compute2D(bufferIn, W, H, Z, NbNDG, Matrix, DMax, NbNDG, false);

    ComputeFeatures(Matrix, DMax, NbNDG, szFileName);

    delete[] bufferIn;
    delete[] Matrix;

    return RES_OK;
  }

  // distance zone matrix
  template <class T1>
  RES_T Thibault_GLDZM(const Image<T1> &imIn, int NbNDG, int GeodesicMethod,
                       char *szFileName)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();
    int i, DMax;

    typename Image<T1>::lineType pixelsIn = imIn.getPixels();
    UINT8 *bufferIn                       = new UINT8[W * H * Z];

    // Sous echantilonnage
    for (i = W * H * Z - 1; i >= 0; i--)
      bufferIn[i] = (int) round((pixelsIn[i] / 255.0) * NbNDG);

    DMax = (int) Compute2D(bufferIn, W, H, Z, NbNDG, 0, 0, 0, GeodesicMethod);

    int *Matrix = new int[DMax * NbNDG];

    for (i = NbNDG * DMax - 1; i >= 0; i--)
      Matrix[i] = 0;

    Compute2D(bufferIn, W, H, Z, NbNDG, Matrix, DMax, NbNDG, GeodesicMethod);

    ComputeFeatures(Matrix, DMax, NbNDG, szFileName);

    delete[] bufferIn;
    delete[] Matrix;

    return RES_OK;
  }
} // namespace smil

#endif
