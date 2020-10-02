#ifndef __PO_GLSZM_T_HPP__
#define __PO_GLSZM_T_HPP__

#include "Core/include/DCore.h"

namespace smil
{
  template <typename T>
  class ZoneMatrixFunctor
  {
  public:
    ZoneMatrixFunctor()
    {
      Sum = 1.;
    }

    ~ZoneMatrixFunctor()
    {
      Sum = 1.;
    }

  private:
    double Sum;

    // --------------------------------------------
    // Calcul des caracteristiques
    // --------------------------------------------

    /* F0 - Small Zone Emphasis,
     * SZE. Petites zones.
     * @return SZE.
     */
    double SZE(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val += matrix[s + n * Width] / pow(double(s + 1), 2.0);
      return val / Sum;
    }

    /* F1 - Large Zone Emphasis, LZE. Grande zone.
     * @return LZE.
     */
    double LZE(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val += matrix[s + n * Width] * pow(double(s + 1), 2.0);
      return val / Sum;
    }

    /* F2 - Low Gray level Zone Emphasis, LGZE.
     * @return LGZE.
     */
    double LGZE(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val += matrix[s + n * Width] / pow(double(n + 1), 2.0);
      return val / Sum;
    }

    /* F3 - High Gray level Zone Emphasis, HGZE.
     * @return HGZE.
     */
    double HGZE(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val += matrix[s + n * Width] * pow(double(n + 1), 2.0);
      return val / Sum;
    }

    /* F4 - Small Zone Low Gray level Emphasis, SZLGE.
     * @return SZLGE.
     */
    double SZLGE(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val += matrix[s + n * Width] /
                 (pow(double(n + 1), 2.0) * pow(double(s + 1), 2.0));
      return val / Sum;
    }

    /* F5 - Small Zone High Gray level Emphasis, SZHGE.
     * @return SZHGE.
     */
    double SZHGE(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val +=
              matrix[s + n * Width] * pow(double(n + 1) / double(s + 1), 2.0);
      return val / Sum;
    }

    /* F6 - Large Zone Low Gray level Emphasis, LZLGE.
     * @return LSLGLE.
     */
    double LZLGE(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val +=
              matrix[s + n * Width] * pow(double(s + 1) / double(n + 1), 2.0);
      return val / Sum;
    }

    /* F7 - Large Zone High Gray level Emphasis, LZHGE.
     * @return LZHGE.
     */
    double LZHGE(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val += matrix[s + n * Width] * pow(double(n + 1), 2.0) *
                 pow(double(s + 1), 2.0);
      return val / Sum;
    }

    /* F8 - Gray Level Non Uniform, GLNU. Homogeneite spectrale.
     * @return GLNU.
     */
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

    /* F9 - Size Zone Non Uniform, SZNU. Uniformite.
     * @return SZNU.
     */
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

    /* F10 - Zone Percentage, ZPC. Egalite des isotailles (pourcentage
     * primitives).
     * @return ZPC.
     */
    double ZPC(int *matrix, int Width, int nbGrayLevel)
    {
      double val = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          val += double(s + 1) * matrix[s + n * Width];
      return Sum / val;
    }

    /* F11 - Methode qui calcule le barycentre sur les niveaux de gris.

     * @return Le barycentre sur les niveaux de gris.
     */
    double BARYGL(int *matrix, int Width, int nbGrayLevel)
    {
      double mean = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          mean += double(n + 1) * matrix[s + n * Width];
      return mean / Sum;
    }

    /* F12 - Methode qui calcule le barycentre sur les tailles.
     * @return Le barycentre sur les tailles.
     */
    double BARYS(int *matrix, int Width, int nbGrayLevel)
    {
      double mean = 0.0;
      for (int n = 0; n < nbGrayLevel; n++)
        for (int s = 0; s < Width; s++)
          mean += double(s + 1) * matrix[s + n * Width];
      return mean / Sum;
    }

    void ComputeFeatures(int *Matrix, int W, int H, vector<double> &features)
    {
      features.push_back(SZE(Matrix, W, H));
      features.push_back(LZE(Matrix, W, H));
      features.push_back(LGZE(Matrix, W, H));
      features.push_back(HGZE(Matrix, W, H));
      features.push_back(SZLGE(Matrix, W, H));
      features.push_back(SZHGE(Matrix, W, H));
      features.push_back(LZLGE(Matrix, W, H));
      features.push_back(LZHGE(Matrix, W, H));
      features.push_back(GLNU(Matrix, W, H));
      features.push_back(SZNU(Matrix, W, H));
      features.push_back(BARYGL(Matrix, W, H));
      features.push_back(BARYS(Matrix, W, H));
    }

    int ComputeBinaryCC(int *DistanceMap, std::queue<int> *fifoSave, int W,
                        int H, int D, int BaryX, int BaryY, int BaryZ,
                        int method = 0)
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
        Dist = (float) sqrt(double((X - BaryX) * (X - BaryX) +
                                   (Y - BaryY) * (Y - BaryY) +
                                   (Z - BaryZ) * (Z - BaryZ)));

        if (Dist > DistMax) {
          DistMax  = Dist;
          IndStart = currentPixel;
        }
      } while (!fifoCurrent.empty());

      // Get the max distance
      bool NewDist;
      int k, l, j;
      int Ind;
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
        NewDist = false;

        // For all the neighbour
        for (j = -1; j <= 1; j++) {
          if (Z + j >= 0 && Z + j < D) {
            for (k = -1; k <= 1; k++) {
              if (X + k >= 0 && X + k < W) {
                for (l = -1; l <= 1; l++) {
                  if (k == 0 && l == 0 && j == 0)
                    continue;
                  if (Y + l >= 0 && Y + l < H) {
                    Ind = X + k + (Y + l) * W + (Z + j) * W * H;
                    if (DistanceMap[Ind] == -1) {
                      fifoCurrent.push(Ind);
                      DistanceMap[Ind] = -2;
                    } else if (DistanceMap[Ind] != 0 &&
                               DistanceMap[Ind] != -2) {
                      float D = sqrt((float) ((j * j) + (k) * (k) + (l) * (l)));
                      if (!NewDist || Dist > (DistanceMap[Ind] + D)) {
                        Dist    = (float) DistanceMap[Ind] + D;
                        NewDist = true;
                      }
                    }

                    /*

                     *!(FLOAT_EQ_CORRIGE(Dist,DataTraits<F_SIMPLE>::
                     *        default_value::max_value()))){
                     *  ==> Dist != max_value()
                     */
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
                }
              }
            }
          }
        }
      } while (!fifoCurrent.empty());

      // Write on DistanceMap the Max Distance for all the pixel of the CC,
      // we pop fifoSave
      int size = (int) fifoSave->size();
      int res;
      do {
        currentPixel = fifoSave->front();
        fifoSave->pop();
        if (method == 1) {
          res = (int) DistMax;
        } else if (method == 2) {
          // En 2D
          if (D == 1)
            res = (int) (DistMax * DistMax / double(size) * 0.785398);
          // En 3D
          else
            res = (int) (DistMax * DistMax * DistMax / double(size) * 0.523599);

        } else {
          int X2, Y2, Z2;
          X2          = IndStart % W;
          Y2          = (IndStart % (W * H) - X2) / W;
          Z2          = (IndStart - X2 - Y2 * W) / (W * H);
          double Eucl = sqrt(double((Xtort - X2) * (Xtort - X2) +
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

    double ComputeBinary(int *pixelDone, int W, int H, int D, int *Matrix,
                         int mW, SMIL_UNUSED int mH, int Level,
                         int method = 0)
    {
      std::queue<int> fifoCurrent, fifoSave;
      int X, Y, Z, currentPixel, i, j, k, l, Ind;
      int BaryX, BaryY, BaryZ, NbPixel, Max = 0;

      for (i = W * H * D - 1; i >= 0; i--) { // For all pixels
        if (pixelDone[i] == 1) {
          // We are on a CC
          fifoCurrent.push(i);
          if (method)
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
            for (j = -1; j <= 1; j++) {
              if (Z + j >= 0 && Z + j < D) {
                for (k = -1; k <= 1; k++) {
                  if (X + k >= 0 && X + k < W) {
                    for (l = -1; l <= 1; l++) {
                      if (k == 0 && l == 0 && j == 0)
                        continue;
                      if (Y + l >= 0 && Y + l < H) {
                        Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                        if (pixelDone[Ind] == 1) {
                          pixelDone[Ind] = -1;
                          fifoCurrent.push(Ind);
                          if (method)
                            fifoSave.push(Ind);
                        }
                      }
                    }
                  }
                }
              }
            }
          } while (!fifoCurrent.empty());

          if (method != 0) {
            // One CC have been push, we compute it
            BaryX   = (int) (BaryX / fifoSave.size());
            BaryY   = (int) (BaryY / fifoSave.size());
            BaryZ   = (int) (BaryZ / fifoSave.size());
            NbPixel = ComputeBinaryCC(pixelDone, &fifoSave, W, H, D, BaryX,
                                      BaryY, BaryZ, method);
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

    double Compute2D(UINT8 *bufferIn, int W, int H, int Z, int NbNDG,
                     int *Matrix, int mW, int mH, int method = 0)
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
                            method);
          if (M > Max)
            Max = M;
        }
      }

      delete[] pixelDone;
      return Max;
    }

  public:
    template <class T1>
    RES_T Size(const Image<T1> &imIn, int NbNDG, vector<double> &features)
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

      ComputeFeatures(Matrix, DMax, NbNDG, features);

      delete[] bufferIn;
      delete[] Matrix;

      return RES_OK;
    }

    // distance zone matrix
    template <class T1>
    RES_T Distance(const Image<T1> &imIn, int NbNDG, int method,
                   vector<double> &features)
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

      DMax = (int) Compute2D(bufferIn, W, H, Z, NbNDG, 0, 0, 0, method);

      int *Matrix = new int[DMax * NbNDG];

      for (i = NbNDG * DMax - 1; i >= 0; i--)
        Matrix[i] = 0;

      Compute2D(bufferIn, W, H, Z, NbNDG, Matrix, DMax, NbNDG, method);

      ComputeFeatures(Matrix, DMax, NbNDG, features);

      delete[] bufferIn;
      delete[] Matrix;

      return RES_OK;
    }
  };

  /*
   * #    #    #   #####  ######  #####   ######    ##     ####   ######
   * #    ##   #     #    #       #    #  #        #  #   #    #  #
   * #    # #  #     #    #####   #    #  #####   #    #  #       #####
   * #    #  # #     #    #       #####   #       ######  #       #
   * #    #   ##     #    #       #   #   #       #    #  #    #  #
   * #    #    #     #    ######  #    #  #       #    #   ####   ######
   */
  template <class T1>
  vector<double> grayLevelZMSize(const Image<T1> &imIn, int NbNDG)
  {
    ZoneMatrixFunctor<T1> zm;
    vector<double> features;

    zm.Size(imIn, NbNDG, features);
    return features;
  }

  template <class T1>
  vector<double> grayLevelZMDistance(const Image<T1> &imIn, int NbNDG,
                                     int method)
  {
    ZoneMatrixFunctor<T1> zm;
    vector<double> features;

    zm.Distance(imIn, NbNDG, method, features);
    return features;
  }
} // namespace smil

#endif
