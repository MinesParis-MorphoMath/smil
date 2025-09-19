#ifndef __CANNY_FILTER_T_HPP__
#define __CANNY_FILTER_T_HPP__

#include <math.h>

namespace smil
{
//  {
/**
* @author Vincent Morard
* @brief Canny edge detection: Canny's aim was to discover the optimal edge
detection algorithm. In this situation, an "optimal" edge detector means:
*  -good detection - the algorithm should mark as many real edges in the image
as possible.
*  -good localization - edges marked should be as close as possible to the edge
in the real image.
*  -minimal response - a given edge in the image should only be marked once, and
where possible, image noise should not create false edges.

* To satisfy these requirements Canny used the calculus of variations - a
technique which finds the function which optimizes a given functional.
* The optimal function in Canny's detector is described by the sum of four
exponential terms, but can be approximated by the first derivative of a
Gaussian.
* John Canny, A computational approach to edge detection, IEEE Pami, vol. 8, n°
6, novembre 1986, pp 679-698
*/

//****************************************************************************
// Canny.cpp
// Ce fichier regroupe les fonctions permettant de calculer la detection
// optimale des contours suivant la methode instauree par Canny. Le principe est
// le suivant: On applique a l'image source, un filtre passe bas, gaussien
// d'écart type s. Le noyau de convolution sera "Gaussien" et le resultat de la
// convolution sera FlouX et FlouY. En effet le noyau est a 1 dimension donc on
// peut l'appliquer suivant les x et suivant les y.
//
// On cree un second noyau qui sera cette fois-ci la derivee d'une gaussienne :
// DGaussien
// Pour reperer les contours, on applique a l'image FlouX et FlouY le noyau
// DGaussien
// on obtient alors les images dX et dY.
// On supprime alors les maxima de la norme des images dX et dY pour obtenir
// notre image finale.
//
// MAX_SIZE_MASK. On n'autorise pas a avoir un masque de dimension supérieur a
// MAX_SIZE_MASK en effet plus sigma (s) est grand, plus le masque de
// convolution sera grand.
//******************************************************************************
#define MAX_SIZE_MASK 20
#define MAG_SCALE 1
#define ORI_SCALE 1
  // #define PI 3.14159

  //****************************************************************************
  // Norm:
  // Calcul de la norme d'un vecteur
  //****************************************************************************
  inline float Norm(float x, float y)
  {
    return (float) std::sqrt((double) (x * x + y * y));
  }

  //****************************************************************************
  // Gauss:
  // on calcule la valeur en x de la fonction de gauss
  //****************************************************************************
  inline float Gauss(float x, float Sigma)
  {
    return (float) std::exp((double) ((-x * x) / (2 * Sigma * Sigma)));
  }

  //****************************************************************************
  // MeanGauss:
  // On calcule la moyenne de trois valeurs de gauss autour de x
  //****************************************************************************
  inline float MeanGauss(float x, float Sigma)
  {
    float z;
    z = (float) ((Gauss(x, Sigma) + Gauss((float) (x + 0.5), Sigma) +
                  Gauss((float) (x - 0.5), Sigma)) /
                 (float) 3.0);
    return (float) (z / (PI * 2.0 * Sigma * Sigma));
  }

  //****************************************************************************
  // dGauss
  // On calcule ici la valeur de la derivée de la fonction de gauss au point
  // x.
  //****************************************************************************
  inline float dGauss(float x, float Sigma)
  {
    return -x / (Sigma * Sigma) * Gauss(x, Sigma);
  }

  //****************************************************************************
  // Separable convolution
  // Cette fonction permet de convoluer l'image par un masque (gau)
  // Le resultat sera stocke dans les variables GauX et GauY
  //****************************************************************************
  template <typename T1>
  static void SeparableConvolution(T1 *imIn, int Width, int Height, float *gau,
                                   int taille, float *GauX, float *GauY)
  {
    int i, j, k, I1, I2;

    float x, y;

    // Pour tous les pixels
    for (i = 0; i < Width; i++)
      for (j = 0; j < Height; j++) {
        x = (float) (gau[0] * imIn[i + j * Width]);
        y = x;
        for (k = 1; k < taille; k++) // Pour tous les pixels du noyau
        {
          I1 = (i + k) % Width;
          I2 = (i - k + Width) % Width;
          if (I1 >= 0 && I1 < Width && I2 >= 0 && I2 < Width)
            y += (float) (gau[k] * imIn[I1 + j * Width] +
                          gau[k] * imIn[I2 + j * Width]);

          I1 = (j + k) % Width;
          I2 = (j - k + Width) % Width;
          if (I1 >= 0 && I1 < Height && I2 >= 0 && I2 < Height)
            x += (float) (gau[k] * imIn[i + I1 * Width] +
                          gau[k] * imIn[i + I2 * Width]);
        }
        GauX[i + j * Width] = x;
        GauY[i + j * Width] = y;
      }
  }

  //****************************************************************************
  // SeparableConvolution_dxy
  // On calcule le resultat de la convolution de l'image avec un masque
  // Le resultat sera stocker dans la variable appele Derive
  //****************************************************************************
  static void SeparableConvolution_dxy(int Width, int Height, float *Image,
                                       float *gau, int taille, float *Derive,
                                       bool Direction)
  {
    int   i, j, k, I1, I2;
    float x;

    for (i = 0; i < Width; i++)
      for (j = 0; j < Height; j++) {
        x = 0.0;
        for (k = 1; k < taille; k++) {
          if (!Direction) {
            I1 = (i + k) % Width;
            I2 = (i - k + Width) % Width;
            if (I1 >= 0 && I1 < Width && I2 >= 0 && I2 < Width)
              x += -gau[k] * Image[I1 + j * Width] +
                   gau[k] * Image[I2 + j * Width];
          } else {
            I1 = (j + k) % Height;
            I2 = (j - k + Height) % Height;
            if (I1 >= 0 && I1 < Height && I2 >= 0 && I2 < Height)
              x += -gau[k] * Image[i + I1 * Width] +
                   gau[k] * Image[i + I2 * Width];
          }
        }
        Derive[i + j * Width] = x;
      }
  }

  //*************************************************************************
  // On regarde les 2 pixels dans la direction du gradient du pixel central
  // 1 de chaque cote. On garde le pixel central (i) si le gradient de i est
  // supérieur au gradient des 2 autres pixels
  // On effectue une interpolation dans le cas ou le pixel se situant dans la
  // direction du gradient tombe au milieu de 2 pixels.
  //*************************************************************************
  template <typename T2>
  static void NonMaxSuppress(int Width, int Height, float *dX, float *dY,
                             T2 *imNorme)
  {
    int i, j;

    float x, y, N, N1, N2, N3, N4, VectX, VectY;

    for (i = 0; i < Width; i++)
      for (j = 0; j < Height; j++) {
        imNorme[i + j * Width] = 0;
        if (i == 0 || j == 0 || i == Width - 1 || j == Height - 1)
          continue;

        // On considere dx et dy comme des vecteurs
        VectX = dX[i + j * Width];
        VectY = dY[i + j * Width];
        if (fabs(VectX) < 0.01 && fabs(VectY) < 0.01)
          continue;

        N = Norm(VectX, VectY);

        // on suit la direction du gradient qui est indique par les vecteurs
        // VectX et VectY et retient uniquement les pixels qui correspondent
        //a des maximum locaux

        if (fabs(VectY) > fabs(VectX)) // la direction du vecteur est verticale
        {
          x  = fabs(VectX) / fabs(VectY);
          y  = 1;
          N2 = Norm(dX[i - 1 + j * Width], dY[i - 1 + j * Width]);
          N4 = Norm(dX[i + 1 + j * Width], dY[i + 1 + j * Width]);

          if (VectX * VectY > 0) {
            N3 = Norm(dX[i + 1 + (j + 1) * Width], dY[i + 1 + (j + 1) * Width]);
            N1 = Norm(dX[i - 1 + (j - 1) * Width], dY[i - 1 + (j - 1) * Width]);
          } else {
            N3 = Norm(dX[i + 1 + (j - 1) * Width], dY[i + 1 + (j - 1) * Width]);
            N1 = Norm(dX[i - 1 + (j + 1) * Width], dY[i - 1 + (j + 1) * Width]);
          }
        } else // la direction du vecteur est horizontale
        {
          x  = fabs(VectY) / fabs(VectX);
          y  = 1;
          N2 = Norm(dX[i + (j + 1) * Width], dY[i + (j + 1) * Width]);
          N4 = Norm(dX[i + (j - 1) * Width], dY[i + (j - 1) * Width]);

          if (VectX * VectY > 0) {
            N3 = Norm(dX[i - 1 + (j - 1) * Width], dY[i - 1 + (j - 1) * Width]);
            N1 = Norm(dX[i + 1 + (j + 1) * Width], dY[i + 1 + (j + 1) * Width]);
          } else {
            N3 = Norm(dX[i + 1 + (j - 1) * Width], dY[i + 1 + (j - 1) * Width]);
            N1 = Norm(dX[i - 1 + (j + 1) * Width], dY[i - 1 + (j + 1) * Width]);
          }
        }

        // On calcul l'interpolation du gradient
        if (N > (x * N1 + (y - x) * N2) && N > (x * N3 + (y - x) * N4))
          imNorme[i + j * Width] = (T2)(N * MAG_SCALE);
      }
  }

  //*************************************************************************
  // Canny:
  // Cette fonction comporte 3 parametres:
  // ImgNorme : il s'agit de l'image destination: l'image des contours
  // ImgOrientation : Il s'agit de la direction du gradient.(Peu utilisable)
  // atan(dY/dX) Sigma : il s'agit de l'écart type de l'image. Plus s sera
  // grand, plus le filtrage sera fort et moins il y aura de details. Au
  // contraire si s est faible, le filtrage sera faible et on observera plus
  // de contours.
  //*************************************************************************
  template <typename T1, typename T2>
  static RES_T Canny(T1 *imIn, int W, int H, double Sigma, T2 *imNorme)
  {
    float  Gaussien[MAX_SIZE_MASK], dGaussien[MAX_SIZE_MASK];
    int    i;
    float *dX = NULL, *dY = NULL, *FlouX = NULL, *FlouY = NULL;
    int    taille = 0;

    if (imIn == NULL || imNorme == NULL)
      return RES_ERR_BAD_ALLOCATION;

    // Creation du masque gaussian
    for (i = 0; i < MAX_SIZE_MASK; i++) {
      Gaussien[i]  = MeanGauss((float) i, (float) Sigma);
      dGaussien[i] = dGauss((float) i, (float) Sigma);
      if (Gaussien[i] < 0.005) {
        taille = i;
        break;
      }
    }

    // Allocation pour les images floutees. On garde les chiffres apres la
    // virgule
    FlouX = new float[W * H];
    FlouY = new float[W * H];
    if (FlouX == NULL || FlouY == NULL)
      return RES_ERR_BAD_ALLOCATION;

    // On applique le filtre gaussien a l'image pour supprimer le bruit
    // on effectue 1 convolution 2D qui nous permettra de trouver la
    // composante X et Y
    SeparableConvolution(imIn, W, H, Gaussien, taille, FlouX, FlouY);

    // On effectue maintenant la convolution avec la derive d'un filtre
    // gaussien sur l'image precedement obtenue
    dX = new float[W * H];
    dY = new float[W * H];
    if (dX == NULL || dY == NULL) {
      delete[] FlouX;
      delete[] FlouY;
      return RES_ERR_BAD_ALLOCATION;
    }
    SeparableConvolution_dxy(W, H, FlouX, dGaussien, taille, dX, true);
    SeparableConvolution_dxy(W, H, FlouY, dGaussien, taille, dY, false);

    // deallocation de la memoire
    delete[] FlouX;
    delete[] FlouY;

    // Suppression des non maxima : on ne garde que les maxima locaux
    NonMaxSuppress(W, H, dX, dY, imNorme);

    delete[] dX;
    delete[] dY;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T cannyEdgeDetection(const Image<T1> &imIn, const double Sigma,
                           Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t s[3];
    imIn.getSize(s);

    // TODO: check that image is 2D
    if (s[2] > 1) {
      // Error : this is a 3D image
    }

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    return Canny(bufferIn, s[0], s[1], Sigma, bufferOut);
  }

} // namespace smil
#endif
