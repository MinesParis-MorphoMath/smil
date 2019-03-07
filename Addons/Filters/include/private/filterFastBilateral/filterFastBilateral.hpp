#ifndef __FAST_BILATERAL_FILTER_T_HPP__
#define __FAST_BILATERAL_FILTER_T_HPP__

#include <math.h>
#include <complex>
#include "Core/include/DCore.h"

using namespace std;

namespace smil
{
  template <class Tin>
  static void _fastBilateralFilter(Tin *T, int W, int H, int Z, int methodS,
                            int methodG, int nS, double EcTSx, double EcTSy,
                            double EcTSz, double EcTGx, double EcTGy,
                            double EcTGz)
  // Description: filtrage par filtre bilateral rapide, version non exacte
  // Param: T(E) Vint 3D
  // Param: W,H,Z(E) taille Vint
  // Param: methodS,methodG(E) estimator 1->Gauss, !1->Tukey pour filtrage
  // spatial S et niveau de gris G Param: nS(E) taille voisinage (à l'extérieur,
  // Gaussienne nulle) Param: EcTSx,y,z(E) ecart type Gaussienne filtrage
  // spatial suivant axes x, y et z Param: EcTGx,y,z(E) Ecart type Gaussienne
  // filtrage niveau de gris suivant axes x, y et z Retour: Vint 3D Remarque:
  // calcul inexact (car le filtre n'est pas décomposable), mais les résultats
  // sont corrects et beaucoup plus rapide Possibilité de gérer différement le
  // filtrage suivant les axes valeur typique EcTS=3ou5 EcTG=20ou40 Equivalence
  // Gauss - Tukey: Ect - Ect*Rac(5) Exemple de paramètres correct:
  // D=FastBilateralFilter(D,W,H,Z,2,2,10,3,3,3,20,20,20);
  // ou
  // D=FastBilateralFilter(D,W,H,Z,2,2,10,5,5,5,20,20,20); (un peu plus lent
  // mais resultat encore meilleurs)
  {
    double kk;
    int i, j, k, x, y, z;
    double val1, val2;
    double TI;

    if (methodS != 1) // Tukey correction (pour correspondre à un écart type de
                      // Gaussienne, plus intuitif)
    {
      EcTSx *= sqrt(5.0);
      EcTSy *= sqrt(5.0);
      EcTSz *= sqrt(5.0);
      EcTGx *= sqrt(5.0);
      EcTGy *= sqrt(5.0);
      EcTGz *= sqrt(5.0);
    }

    // Recalcul nS si trop grand.

    double prec = 0.2; // en dessous de prec, on coupe le noyau
    int nSx = nS, nSy = nS, nSz = nS;
    if (methodS == 1) // Gauss
    {
      for (i = nS; i >= 0; i--) {
        if (exp(-0.5 * ((double) i / EcTSx) * ((double) i / EcTSx)) < prec)
          nSx = i;
        if (exp(-0.5 * ((double) i / EcTSy) * ((double) i / EcTSy)) < prec)
          nSy = i;
        if (exp(-0.5 * ((double) i / EcTSz) * ((double) i / EcTSz)) < prec)
          nSz = i;
      }
    } else // Tukey
    {
      for (i = nS; i >= 0; i--) {
        val1 = (double) i / EcTSx;
        if ((0.5 * (1 - val1 * val1) * (1 - val1 * val1)) < prec && i < EcTSx)
          nSx = i;
        val1 = (double) i / EcTSy;
        if ((0.5 * (1 - val1 * val1) * (1 - val1 * val1)) < prec && i < EcTSy)
          nSy = i;
        val1 = (double) i / EcTSz;
        if ((0.5 * (1 - val1 * val1) * (1 - val1 * val1)) < prec && i < EcTSz)
          nSz = i;
      }
    }

    // Calcul LUT precalcul:
    int M = 256;

    double *PrecalCx = new double[M];
    double *PrecalSx = new double[M];
    double *PrecalCy = new double[M];
    double *PrecalSy = new double[M];
    double *PrecalCz = new double[M];
    double *PrecalSz = new double[M];

    for (i = 0; i < M; i++) {
      val1 = (double) i / EcTSx;
      val2 = (double) i / EcTGx;
      if (methodS == 1)
        PrecalSx[i] = exp(-0.5 * val1 * val1);
      else if (i <= EcTSx)
        PrecalSx[i] = 0.5 * (1 - val1 * val1) * (1 - val1 * val1);
      else
        PrecalSx[i] = 0;
      if (methodG == 1)
        PrecalCx[i] = exp(-0.5 * val2 * val2);
      else if (i <= EcTGx)
        PrecalCx[i] = 0.5 * (1 - val2 * val2) * (1 - val2 * val2);
      else
        PrecalCx[i] = 0;
      val1 = (double) i / EcTSy;
      val2 = (double) i / EcTGy;
      if (methodS == 1)
        PrecalSy[i] = exp(-0.5 * val1 * val1);
      else if (i <= EcTSy)
        PrecalSy[i] = 0.5 * (1 - val1 * val1) * (1 - val1 * val1);
      else
        PrecalSy[i] = 0;
      if (methodG == 1)
        PrecalCy[i] = exp(-0.5 * val2 * val2);
      else if (i <= EcTGy)
        PrecalCy[i] = 0.5 * (1 - val2 * val2) * (1 - val2 * val2);
      else
        PrecalCy[i] = 0;
      val1 = (double) i / EcTSz;
      val2 = (double) i / EcTGz;
      if (methodS == 1)
        PrecalSz[i] = exp(-0.5 * val1 * val1);
      else if (i <= EcTSz)
        PrecalSz[i] = 0.5 * (1 - val1 * val1) * (1 - val1 * val1);
      else
        PrecalSz[i] = 0;
      if (methodG == 1)
        PrecalCz[i] = exp(-0.5 * val2 * val2);
      else if (i <= EcTGz)
        PrecalCz[i] = 0.5 * (1 - val2 * val2) * (1 - val2 * val2);
      else
        PrecalCz[i] = 0;
    }

    // Tukey:
    if (methodS != 1) {
      if (nSx > EcTSx)
        nSx = (int) EcTSx;
      if (nSy > EcTSy)
        nSy = (int) EcTSy;
      if (nSz > EcTSz)
        nSz = (int) EcTSz;
    }

    double *Tbuf;
    if (Z > 1) {
      Tbuf = new double[Z];
      for (j = 0; j < H; j++) {
        for (i = 0; i < W; i++) {
          for (k = 0; k < Z; k++)
            Tbuf[k] = T[k * W * H + j * W + i];
          for (k = 0; k < Z; k++) {
            kk = 0;
            TI = 0;
            // JOE A voir... gcc gives a warning
            // Line T[k*W*H... is inside or outside for loop ?
            for (z = -nSz; z <= nSz; z++) {
              if ((k + z) >= 0 && (k + z) < Z) {
                val1 = PrecalSz[size_t(abs(z))] *
                       PrecalCz[(int) abs(Tbuf[k + z] - Tbuf[k])];
                kk += val1;
                TI += val1 * (double) Tbuf[k + z];
              }
            }
            T[k * W * H + j * W + i] = (Tin)(TI / kk);
          }
        }
      }
      delete[] Tbuf;
    }

    Tbuf = new double[H];
    for (k = 0; k < Z; k++) {
      for (i = 0; i < W; i++) {
        for (j = 0; j < H; j++)
          Tbuf[j] = T[k * W * H + j * W + i];
        for (j = 0; j < H; j++) {
          kk = 0;
          TI = 0;
          // JOE A voir... gcc gives a warning
          // Line T[k*W*H... is inside or outside for loop ?
          for (y = -nSy; y <= nSy; y++) {
            if ((j + y) >= 0 && (j + y) < H) {
              val1 = PrecalSy[size_t(abs(y))] *
                     PrecalCy[(int) abs(Tbuf[j + y] - Tbuf[j])];
              kk += val1;
              TI += val1 * (double) Tbuf[j + y];
            }
          }
          T[k * W * H + j * W + i] = (Tin)(TI / kk);
        }
      }
    }
    delete[] Tbuf;

    Tbuf = new double[W];
    for (k = 0; k < Z; k++) {
      for (j = 0; j < H; j++) {
        for (i = 0; i < W; i++)
          Tbuf[i] = T[k * W * H + j * W + i];
        for (i = 0; i < W; i++) {
          kk = 0;
          TI = 0;
          // JOE A voir... gcc gives a warning
          // Line T[k*W*H... is inside or outside for loop ?
          for (x = -nSx; x <= nSx; x++) {
            if ((i + x) >= 0 && (i + x) < W) {
              val1 = PrecalSx[size_t(abs(x))] *
                     PrecalCx[(int) abs(Tbuf[i + x] - Tbuf[i])];
              kk += val1;
              TI += val1 * (double) Tbuf[i + x];
            }
          }
          T[k * W * H + j * W + i] = (Tin)(TI / kk);
        }
      }
    }
    delete[] Tbuf;
    delete[] PrecalCx;
    delete[] PrecalSx;
    delete[] PrecalCy;
    delete[] PrecalSy;
    delete[] PrecalCz;
    delete[] PrecalSz;
  }

  template <class T1, class T2>
  RES_T ImFastBilateralFilter(const Image<T1> &imIn, UINT8 Method, UINT8 nS,
                              UINT32 EctS, UINT32 EctG, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    RES_T res = copy(imIn, imOut);
    if (res != RES_OK)
      return res;
    size_t s[3];
    imIn.getSize(s);

    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    _fastBilateralFilter(bufferOut, s[0], s[1], s[2], Method, Method, nS, EctS,
                         EctS, EctS, EctG, EctG, EctG);

    return RES_OK;
  }

  /*template<class T1, class T2>
    RES_C  ImFastBilateralFilterRGB(const Image<T1> & imIn,
    const UINT8 Method,
    const UINT8 nS,
    const UINT32 EctS,
    const UINT32 EctG,
    Image<T2> & imOut)
  {

    RES_C res;

    // Check inputs
    if( ! imIn.isAllocated() || ! imOut.isAllocated() ){
      MORPHEE_REGISTER_ERROR("Image not allocated");
      return RES_NOT_ALLOCATED;
    }
    if(!t_CheckWindowSizes(imIn, imOut)){
      MORPHEE_REGISTER_ERROR("Bad window sizes");
      return RES_ERROR_BAD_WINDOW_SIZE;
    }

    // Initialisation
    typedef typename T1::value_type Tin;
    morphee::Image<Tin> imInR = imIn.template t_getSame<Tin>();
    morphee::Image<Tin> imInG = imIn.template t_getSame<Tin>();
    morphee::Image<Tin> imInB = imIn.template t_getSame<Tin>();

    // Get the 3 channels
    res = t_colorSplitTo3(imIn,imInR,imInG,imInB);
    if(res != RES_OK){
      MORPHEE_REGISTER_ERROR("Error t_colorSplitTo3() in
  t_ImFastBilateralFilterRGB "); return RES_ERROR;
    }

    int W,H,Z;
    W=imIn.getWxSize();
    H=imIn.getWySize();
    Z=imIn.getWzSize();

    //
  ****************************************************************************************
    // Computation of the R,G,B channel
    // Fast bilateral filter
    FastBilateralFilter(imInR.rawPointer(),W,H,Z,Method,Method,nS,EctS,EctS,EctS,EctG,EctG,EctG);
    FastBilateralFilter(imInG.rawPointer(),W,H,Z,Method,Method,nS,EctS,EctS,EctS,EctG,EctG,EctG);
    FastBilateralFilter(imInB.rawPointer(),W,H,Z,Method,Method,nS,EctS,EctS,EctS,EctG,EctG,EctG);
    //
  ****************************************************************************************




    // Combinaision of the 3 channels
    res = t_colorComposeFrom3(imInR,imInG,imInB,imOut);
    if(res!=RES_OK){
      MORPHEE_REGISTER_ERROR("Error colorComposeFrom3() in
  t_ImFastBilateralFilterRGB "); return RES_ERROR;
    }


    return RES_OK;
    }*/

} // namespace smil

#endif
