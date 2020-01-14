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
 *     Integrated into Smil Advanced Source Tree with some cosmetics
 *
 * __HEAD__ - Stop here !
 */


#ifndef __PO_UTILITIES_T_HPP__
#define __PO_UTILITIES_T_HPP__

#include "Core/include/DCore.h"

namespace smil
{
  int GetNbCC(UINT8 *pixelDone, int W, int H, int D)
  {
    std::queue<int> currentQueue;
    int j, k, l, X, Y, Z, currentPixel, Ind;
    int Cpt = 0;
    for (int i = W * H * D - 1; i >= 0; i--) {
      if (pixelDone[i] == 1) {
        currentQueue.push(i);
        pixelDone[i] = 2;
        // We push all the CC
        do {
          currentPixel = currentQueue.front();
          currentQueue.pop();
          X = currentPixel % W;
          Y = (currentPixel % (W * H) - X) / W;
          Z = (currentPixel - X - Y * W) / (W * H);

          // For all the neighbors
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      if (pixelDone[Ind] == 1) {
                        pixelDone[Ind] = 2;
                        currentQueue.push(Ind);
                      }
                    }
        } while (!currentQueue.empty());
        Cpt++;
      }
    }
    return Cpt;
  }

  template <class T>
  RES_T CountNbCCperThreshold(const Image<T> &imIn, int *NbCC, int Invert)
  {
    ASSERT_ALLOCATED(&imIn)

    int W            = imIn.getWidth();
    int H            = imIn.getHeight();
    int Z            = imIn.getDepth();
    int NbWhitePixel = 0;

    int i;
    ULONG Level;
    UINT8 *pixelDone = new UINT8[W * H * Z];
    if (pixelDone == 0) {
      // MORPHEE_REGISTER_ERROR("Error allocation");
      return RES_ERR_BAD_ALLOCATION;
    }
    for (i = 0; i < 256; i++)
      NbCC[i] = (Invert ? 1 : 0);

    typename Image<T>::lineType pixelsIn = imIn.getPixels();

    for (Level = 0; Level < 255; Level++) {
      NbWhitePixel = 0;

      for (size_t i = 0; i < imIn.getPixelCount(); ++i) {
        if (pixelsIn[i] >= Level) {
          if (!Invert)
            pixelDone[i] = 1;
          else
            pixelDone[i] = 0;
          NbWhitePixel++;
        } else {
          if (!Invert)
            pixelDone[i] = 0;
          else
            pixelDone[i] = 1;
        }
      }
      NbCC[Level] = GetNbCC(pixelDone, W, H, Z);
      if (NbWhitePixel == 0)
        break;
      if (Invert && NbCC[Level] == 1 && Level > 10)
        break;
    }
    delete[] pixelDone;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImThresholdWithUniqueCCForBackGround(const Image<T1> &imIn,
                                             Image<T2> &imOut, int sliceBySlice)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    int i, j, Ind, Slice, Threshold[256];
    ULONG Level;

    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    if (sliceBySlice == 0) {
      int NbCC[256];
      CountNbCCperThreshold(imIn, NbCC, 1);
      for (i = 2; i < 256; i++)
        if (NbCC[i] == 1)
          break;
      Threshold[0] = i;
    } else {
      UINT8 *pixelDone = new UINT8[W * H];
      if (pixelDone == 0) {
        // MORPHEE_REGISTER_ERROR("Error allocation");
        return RES_ERR_BAD_ALLOCATION;
      }

      for (Slice = 0; Slice < Z; Slice++) {
        for (Level = 0; Level < 256; Level++) {
          Ind = Slice * W * H;
          for (j = 0, i = Ind; j < W * H; ++j, ++i)
            pixelDone[j] = ((pixelsIn[i] >= Level) ? (0) : (1));
          if (GetNbCC(pixelDone, W, H, 1) == 1)
            break;
        }
        Threshold[Slice] = Level;

        // FILE *fic = fopen("C:\\T1\\CourbeDeSeuil.xls","a");
        // if(fic==0) continue;
        // fprintf(fic,"%i\n",Level);
        // fclose(fic);
      }
      delete[] pixelDone;
    }

    ULONG T, x, y, z;

    for (size_t i = 0; i < imOut.getPixelCount(); ++i) {
      if (sliceBySlice == 0)
        T = Threshold[0];
      else {
        x = i % W;
        y = (i % (W * H) - x) / W;
        z = (i - x - y * W) / (W * H);
        T = Threshold[z];
      }
      pixelsOut[i] = ((pixelsIn[i] <= T) ? (0) : (255));
    }

    return RES_OK;
  }

  template <class T1, class T2>
  RES_T PseudoPatternSpectrum(const Image<T1> &imCont, const Image<T2> &imInd,
                              int *patternSpect)
  {
    ASSERT_ALLOCATED(&imCont)
    ASSERT_ALLOCATED(&imInd)
    ASSERT_SAME_SIZE(&imCont, &imInd)

    // Initialization: 256 for 8 bits
    int i;
    for (i = 0; i < 256; i++)
      patternSpect[i] = 0;

    typename Image<T1>::lineType pixelsCont = imCont.getPixels();
    typename Image<T2>::lineType pixelsInd  = imInd.getPixels();

    for (size_t i = 0; i < imCont.getPixelCount(); ++i)
      patternSpect[pixelsInd[i]] += pixelsCont[i];

    return RES_OK;
  }

  // Parcours des composantes connexes et leur suppression si le nombre de pixel
  // de la CC ne correspond pas au niveau de gris de la CC: (Pour traiter
  // l'indicatrice ou le niveau de gris correspond à la longueur de la fibre.)
  template <class T1, class T2>
  RES_T ImSupSmallRegion(const Image<T1> &imIn, Image<T2> &imOut,
                         float percentage)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    int i, j, x, y, z, k, l, m, nbPixel, W, H, Z;
    ULONG NDG;
    W = imIn.getWidth();
    H = imIn.getHeight();
    Z = imIn.getDepth();

    UINT8 *pixelDone = new UINT8[W * H * Z];
    // JOE 0 or NULL ???
    if (pixelDone == NULL) {
      // MORPHEE_REGISTER_ERROR("Error allocation");
      return RES_ERR_BAD_ALLOCATION;
    }
    std::memset(pixelDone, 0, W * H * Z);

    ImageFreezer freeze(imOut);
    fill(imOut, (T2) 0);
    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    // FIFO to flood the connexes componantes
    std::queue<int> fifoCurrent, fifoSave;

    // For all the pixel of the picture
    for (i = Z * H * W - 1; i >= 0; i--) {
      // Si le pixel n'a pas déjà été traite
      if (pixelDone[i] == 0 && ((NDG = pixelsIn[i]) != 0)) {
        pixelDone[i] = 1;
        fifoCurrent.push(i);
        fifoSave.push(i);
        nbPixel = 1;

        // On empile tous les pixels connexes qui on le niveaude gris NDG
        do {
          // On dépile
          j = fifoCurrent.front();
          fifoCurrent.pop();
          x = j % W;
          y = (j % (W * H) - x) / W;
          z = (j - x - y * W) / (W * H);

          // For all the neighbors
          for (m = -1; m <= 1; m++)
            if (z + m >= 0 && z + m < Z)
              for (k = -1; k <= 1; k++)
                if (x + k >= 0 && x + k < W)
                  for (l = -1; l <= 1; l++) {
                    j = x + k + (y + l) * W + (z + m) * W * H;
                    if (y + l >= 0 && y + l < H && pixelDone[j] == 0 &&
                        pixelsIn[j] == NDG) {
                      nbPixel++;
                      pixelDone[j] = 1;
                      fifoCurrent.push(j);
                      fifoSave.push(j);
                    }
                  }
        } while (!fifoCurrent.empty());

        // In fifoSave, with have all the pixel of the CC
        while (!fifoSave.empty()) {
          j = fifoSave.front();
          fifoSave.pop();

          // Verification du critere de conservation de la CC
          if (nbPixel < NDG * percentage) {
            // We delete the CC
            pixelsOut[j] = 0;
          }
        }
      }
    }
    delete[] pixelDone;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T measComputeVolume(const Image<T1> &imIn, const Image<T2> &imLevel,
                          float *Value)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_ALLOCATED(&imLevel)
    ASSERT_SAME_SIZE(&imIn, &imLevel)

    typename Image<T1>::lineType pixelsIn    = imIn.getPixels();
    typename Image<T2>::lineType pixelsLevel = imLevel.getPixels();
    float signalTot                          = 0;
    *Value                                   = 0;

    for (size_t i = 0; i < imLevel.getPixelCount(); ++i) {
      *Value += pixelsIn[i] * pixelsLevel[i];
      // if(*itLevel!=0) signalTot += (*itIn);
      signalTot += pixelsIn[i];
    }
    *Value = (*Value) / signalTot;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T measComputeIndFromPatternSpectrum(const Image<T1> &imTrans,
                                          const Image<T2> &imInd,
                                          UINT16 BorneMin, UINT16 BorneMax,
                                          UINT8 Normalized, float *Value)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imTrans)
    ASSERT_ALLOCATED(&imInd)
    ASSERT_SAME_SIZE(&imTrans, &imInd)

    // Initialization: 256 for 8 bits
    int Granulo[256] = {0};
    int i;

    typename Image<T1>::lineType pixelsTrans = imTrans.getPixels();
    typename Image<T2>::lineType pixelsInd   = imInd.getPixels();

    for (size_t i = 0; i < imTrans.getPixelCount(); ++i)
      Granulo[pixelsInd[i]] += pixelsTrans[i];

    int Norm = 0;
    *Value   = 0;
    for (i = 0; i < 100; i++) {
      if (i >= BorneMin && i <= BorneMax)
        *Value = (*Value) + Granulo[i];
      if (i > 0)
        Norm += Granulo[i];
    }

    if (Norm == 0) {
      *Value = 0;
      return RES_OK;
    }
    if (Normalized)
      *Value = *Value / (float) Norm;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImThresholdWithMuAndSigma(const Image<T1> &imIn, Image<T2> &imOut,
                                  float paramSigma)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int i, j, Ind, Slice;
    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();
    float Mu, Sigma;

    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    for (Slice = 0; Slice < Z; Slice++) {
      Mu    = 0;
      Sigma = 0;
      Ind   = Slice * W * H;
      // Compute Mu
      for (i = 0, j = Ind; i < W * H; ++j, i++)
        Mu += pixelsIn[j];
      Mu /= (float) (W * H);

      // Compute Sigma
      for (i = 0, j = Ind; i < W * H; ++j, i++)
        Sigma += (pixelsIn[j] - Mu) * (pixelsIn[j] - Mu);
      Sigma /= (float) (W * H);
      Sigma = sqrt(Sigma);

      // Write Output
      for (i = 0, j = Ind; i < W * H; ++j, i++)
        pixelsOut[j] = ((pixelsIn[j] > Mu + paramSigma * Sigma) ? (255) : (0));

      // FILE *fic = fopen("C:\\T1\\CourbeDeSeuil.xls","a");
      // if(fic==0) continue;
      // fprintf(fic,"%f\t%f\t%i\n",Mu,Sigma,(int) (Mu + paramSigma*Sigma));
      // fclose(fic);
    }

    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImElongationFromSkeleton(const Image<UINT8> &imBin,
                                 const Image<T1> &imSk, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imBin)
    ASSERT_ALLOCATED(&imSk)
    ASSERT_SAME_SIZE(&imBin, &imSk)
    ASSERT_SAME_SIZE(&imBin, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int i, j, k, l, X, Y, Z, Ind, currentPixel, cptSk, cptBin;
    int W         = imBin.getWidth();
    int H         = imBin.getHeight();
    int D         = imBin.getDepth();
    int *bufferIn = new int[W * D * H];
    if (bufferIn == 0) {
      // MORPHEE_REGISTER_ERROR("Error allocation bufferIn");
      return RES_ERR_BAD_ALLOCATION;
    }

    std::queue<int> currentQueue, saveQueue;

    typename Image<UINT8>::lineType pixelsBin = imBin.getPixels();
    typename Image<T1>::lineType pixelsSk     = imSk.getPixels();
    typename Image<T2>::lineType pixelsOut    = imOut.getPixels();

    for (size_t i = 0; i < imBin.getPixelCount(); ++i) {
      if (pixelsBin[i] != 0)
        bufferIn[i] = ((pixelsSk[i] != 0) ? (-2) : (-1));
      else
        bufferIn[i] = 0;
    }

    for (i = W * H * D; i >= 0; i--) {
      if (bufferIn[i] == -1 || bufferIn[i] == -2) {
        currentQueue.push(i);
        saveQueue.push(i);

        cptSk  = 0;
        cptBin = 0;

        if (bufferIn[i] == -2)
          cptSk++;
        cptBin++;

        bufferIn[i] = -3;

        do {
          currentPixel = currentQueue.front();
          currentQueue.pop();

          X = currentPixel % W;
          Y = (currentPixel % (W * H) - X) / W;
          Z = (currentPixel - X - Y * W) / (W * H);

          // For all the neigbour
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      if (bufferIn[Ind] == -1 || bufferIn[Ind] == -2) {
                        if (bufferIn[Ind] == -2)
                          cptSk++;
                        cptBin++;
                        bufferIn[Ind] = -3;
                        currentQueue.push(Ind);
                        saveQueue.push(Ind);
                      }
                    }

        } while (!currentQueue.empty());

        while (!saveQueue.empty()) {
          currentPixel = saveQueue.front();
          saveQueue.pop();
          bufferIn[currentPixel] = (int) ((cptSk * cptSk / (float) cptBin));
        }
      }

      for (size_t i = 0; i < imOut.getPixelCount(); ++i)
        pixelsOut[i] = bufferIn[i];
    }
    delete[] bufferIn;
    return RES_OK;
  }

  // Function Hue_2_RGB
  double Hue_2_RGB(double v1, double v2, double vH)
  {
    if (vH < 0)
      vH += 1;
    if (vH > 1)
      vH -= 1;
    if ((6 * vH) < 1)
      return (v1 + (v2 - v1) * 6 * vH);
    if ((2 * vH) < 1)
      return (v2);
    if ((3 * vH) < 2)
      return (v1 + (v2 - v1) * ((0.666666) - vH) * 6);
    return v1;
  }

  void HSL_to_RGB2(double H, double S, double L, double *R, double *G,
                   double *B)
  {
    double tmp1, tmp2;
    // h,s,l compris entre 0 et 1
    H = (double) H / 239;
    S = (double) S / 240;
    L = (double) L / 240;

    if (S == 0) {
      *R = 255 * L;
      *G = 255 * L;
      *B = 255 * L;
    } else {
      if (L < 0.5)
        tmp2 = L * (1 + S);
      else
        tmp2 = (L + S) - (L * S);
      tmp1 = 2 * L - tmp2;

      *R = 255 * Hue_2_RGB(tmp1, tmp2, H + (0.333333));
      *G = 255 * Hue_2_RGB(tmp1, tmp2, H);
      *B = 255 * Hue_2_RGB(tmp1, tmp2, H - (0.333333));
    }
  }

  template <class T>
  RES_T ImFalseColorHSL(const Image<T> &imIn, Image<RGB> &imOut, float Scale)
  {
    return RES_ERR_NOT_IMPLEMENTED;

    /*
//Check inputs
if( ! imIn.isAllocated() || ! imOut.isAllocated()){
  MORPHEE_REGISTER_ERROR("Image not allocated");
  return RES_NOT_ALLOCATED;
}
if(!t_CheckWindowSizes(imIn, imOut)){
  MORPHEE_REGISTER_ERROR("Bad window sizes");
  return RES_ERROR_BAD_WINDOW_SIZE;
}

morphee::Image<morphee::UINT8> imInR = imIn.template t_getSame<morphee::UINT8>
(); morphee::Image<morphee::UINT8> imInG = imIn.template
t_getSame<morphee::UINT8> (); morphee::Image<morphee::UINT8> imInB =
imIn.template t_getSame<morphee::UINT8> ();

typename Image<T1>::const_iterator itIn,itInEnd;
typename Image<T1>::iterator itInR;
typename Image<T1>::iterator itInG;
typename Image<T1>::iterator itInB;
itIn = imIn.begin();
itInEnd = imIn.end();
itInR = imInR.begin();
itInG = imInG.begin();
itInB = imInB.begin();
double R,G,B;
for(;itIn != itInEnd; ++itIn,++itInG,++itInB,++itInR){
  double H = (*itIn)*Scale;
  if(H>240)H=240;
  

  HSL_to_RGB2((255-H)/255.0*200.0,230,108,&R,&G,&B);
  

  if((*itIn)==0){
    *itInR = (UINT8) 0;
    *itInG = (UINT8) 0;
    *itInB = (UINT8) 0;
  }
  else{
    *itInR = (UINT8) R;
    *itInG = (UINT8) G;
    *itInB = (UINT8) B;
  }
}
return morphee::t_colorComposeFrom3(imInR,imInG,imInB,imOut);*/
  }

  void SupTriplePoint(UINT8 *bufferIn, int W, int H, int D,
                      std::vector<int> &currentCC, std::queue<int> *triplePoint)
  {
    int i, j, k, l, currentPixel, nbPixel, X, Y, Z, Ind, nbNeighbor;
    int sizeCC = currentCC.size();
    int size   = sizeCC;

    std::queue<int> currentQueue;
    std::queue<int> potentialTriplePoint;

    for (i = 0; i < sizeCC; i++)
      if (bufferIn[currentCC[i]] != 0)
        bufferIn[currentCC[i]] = 3;

    // On inspect tous les pseudos points triples trouvés et on les supprimes si
    // ce sont des pixels inutils
    while (!triplePoint->empty()) {
      currentPixel = triplePoint->front();
      triplePoint->pop();
      bufferIn[currentPixel] = 0;
      size--;

      // Find the existing init point
      for (i = 0; i < sizeCC; i++)
        if (bufferIn[currentCC[i]] != 0) {
          Ind = currentCC[i];
          break;
        }

      currentQueue.push(Ind);
      bufferIn[Ind] = 4;
      nbPixel       = 1;
      do {
        i = currentQueue.front();
        currentQueue.pop();
        X = i % W;
        Y = (i % (W * H) - X) / W;
        Z = (i - X - Y * W) / (W * H);

        // For all the neighbors
        for (j = -1; j <= 1; j++)
          if (Z + j >= 0 && Z + j < D)
            for (k = -1; k <= 1; k++)
              if (X + k >= 0 && X + k < W)
                for (l = -1; l <= 1; l++)
                  if (Y + l >= 0 && Y + l < H && (k != 0 || l != 0 || j != 0)) {
                    Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                    if (bufferIn[Ind] == 3) {
                      currentQueue.push(Ind);
                      bufferIn[Ind] = 4;
                      nbPixel++;
                    }
                  }
      } while (!currentQueue.empty());

      // If we cannot reconstruct all the skeleton
      if (nbPixel != size) {
        size++;
        bufferIn[currentPixel] = 3;
        // C'est un potentiel point triple!
        potentialTriplePoint.push(currentPixel);
      }

      for (i = 0; i < sizeCC; i++)
        if (bufferIn[currentCC[i]] != 0)
          bufferIn[currentCC[i]] = 3;
    }

    // We check if it really is a triple point. If so, we delete it.
    while (!potentialTriplePoint.empty()) {
      i = potentialTriplePoint.front();
      potentialTriplePoint.pop();
      X          = i % W;
      Y          = (i % (W * H) - X) / W;
      Z          = (i - X - Y * W) / (W * H);
      nbNeighbor = 0;

      // For all the neighbors
      for (j = -1; j <= 1; j++)
        if (Z + j >= 0 && Z + j < D)
          for (k = -1; k <= 1; k++)
            if (X + k >= 0 && X + k < W)
              for (l = -1; l <= 1; l++)
                if (Y + l >= 0 && Y + l < H && (k != 0 || l != 0 || j != 0)) {
                  Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                  if (bufferIn[Ind] != 0)
                    nbNeighbor++;
                }
      if (nbNeighbor >= 3)
        // we delete it!
        bufferIn[i] = 0;
    }
  }

  template <class T1, class T2>
  RES_T ImFromSkeletonSupTriplePoint(const Image<T1> &imIn, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int i, j, k, l, X, Y, Z, NbNeighbor, currentPixel, Ind;
    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int D = imIn.getDepth();

    UINT8 *bufferIn = new UINT8[W * H * D];
    if (bufferIn == 0) {
      // MORPHEE_REGISTER_ERROR("Error allocation bufferIn");
      return RES_ERR_BAD_ALLOCATION;
    }

    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    for (size_t i = 0; i < imIn.getPixelCount(); ++i)
      bufferIn[i] = ((pixelsIn[i] != 0) ? (1) : (0));

    std::queue<int> currentQueue;
    std::vector<int> saveCC;
    std::queue<int> triplePoint;

    for (i = W * H * D; i >= 0; i--)
      if (bufferIn[i] == 1) {
        currentQueue.push(i);
        saveCC.push_back(i);
        bufferIn[i] = 2;

        do {
          currentPixel = currentQueue.front();
          currentQueue.pop();
          X = currentPixel % W;
          Y = (currentPixel % (W * H) - X) / W;
          Z = (currentPixel - X - Y * W) / (W * H);

          NbNeighbor = 0;

          // For all the neighbors
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      if (bufferIn[Ind] == 1) {
                        bufferIn[Ind] = 2;
                        currentQueue.push(Ind);
                        saveCC.push_back(Ind);
                        NbNeighbor++;
                      } else if (bufferIn[Ind] == 2)
                        NbNeighbor++;
                    }
          if (NbNeighbor >= 2)
            triplePoint.push(currentPixel);
        } while (!currentQueue.empty());

        SupTriplePoint(bufferIn, W, H, D, saveCC, &triplePoint);
        saveCC.resize(0);
      }

    for (size_t i = 0; i < imOut.getPixelCount(); ++i)
      pixelsOut[i] = ((bufferIn[i] == 0) ? (0) : (255));

    delete[] bufferIn;
    return RES_OK;
  }

  // Supprime tous les points triples!
  // il faut que la fonction de squelettisation/amincissement produise
  template <class T1, class T2>
  RES_T ImFromSkeletonSupTriblePointBis(const Image<T1> &imIn, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int i, j, k, l, X, Y, Z, NbNeighbor;
    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int D = imIn.getDepth();

    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    std::queue<int> triplePoint;
    // For every pixel
    for (i = 0, Z = 0; Z < D; Z++)
      for (Y = 0; Y < H; Y++)
        for (X = 0; X < W; X++, i++)
          if (pixelsIn[i] != 0) {
            // We are on the Skeleton
            NbNeighbor = 0;
            // For all the neighbors
            for (j = -1; j <= 1; j++)
              if (Z + j >= 0 && Z + j < D)
                for (k = -1; k <= 1; k++)
                  if (X + k >= 0 && X + k < W)
                    for (l = -1; l <= 1; l++)
                      if (Y + l >= 0 && Y + l < H &&
                          (k != 0 || l != 0 || j != 0))
                        if (pixelsIn[(X + k) + (Y + l) * W + (Z + j) * W * H] !=
                            0)
                          // Count the number of neighbors
                          NbNeighbor++;
            if (NbNeighbor >= 3)
              triplePoint.push(i);
          }
    // Suppress triple points
    while (!triplePoint.empty()) {
      i = triplePoint.front();
      triplePoint.pop();
      pixelsOut[i] = 0;
    }

    return RES_OK;
  }

  template <class T>
  RES_T FromSkeletonComputeGranulometry(const Image<T> &imIn, UINT32 *Granulo,
                                        int nbElt, float ScaleX, float ScaleY,
                                        float ScaleZ)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)

    int i, j, k, l, X, Y, Z, NbPixel, currentPixel, Ind;
    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int D = imIn.getDepth();

    for (i = 0; i < nbElt; i++)
      Granulo[i] = 0;

    UINT8 *bufferIn = new UINT8[W * H * D];
    if (bufferIn == 0) {
      // MORPHEE_REGISTER_ERROR("Error allocation bufferIn");
      return RES_ERR_BAD_ALLOCATION;
    }

    typename Image<T>::lineType pixelsIn = imIn.getPixels();

    for (size_t i = 0; i < imIn.getPixelCount(); ++i)
      bufferIn[i] = ((pixelsIn[i] != 0) ? (1) : (0));

    std::queue<int> currentQueue;
    float Dist;
    for (i = W * H * D - 1; i >= 0; i--)
      if (bufferIn[i] == 1) {
        currentQueue.push(i);
        bufferIn[i] = 2;
        NbPixel     = 1;
        Dist        = 1;
        do {
          currentPixel = currentQueue.front();
          currentQueue.pop();
          X = currentPixel % W;
          Y = (currentPixel % (W * H) - X) / W;
          Z = (currentPixel - X - Y * W) / (W * H);

          // For all the neigbour
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      if (bufferIn[Ind] == 1) {
                        Dist += sqrt((float) ((j * ScaleZ * j * ScaleZ) +
                                              (k * ScaleX) * (k * ScaleX) +
                                              (l * ScaleY) * (l * ScaleY)));
                        NbPixel++;
                        bufferIn[Ind] = 2;
                        currentQueue.push(Ind);
                      }
                    }
        } while (!currentQueue.empty());

        // if(NbPixel>=nbElt)NbPixel = nbElt-1;
        // Granulo[NbPixel]++;

        j = (int) Dist;
        if (j > nbElt - 1)
          j = nbElt - 2;
        Granulo[j]++;

        // Last element : Number of connected component
        Granulo[nbElt - 1]++;
      }

    /*
  char Buf[100];
  sprintf(Buf,"C:\\T1\\HistoSKL\\%i.xls",nameFileOut);
  FILE *fic = fopen(Buf,"w");
  nameFileOut++;

  if(fic!=0){
    fprintf(fic,"Taille\tGranulométrie\n");
    for(i=0;i<nbElt;i++)
      fprintf(fic,"%i\t%i\n",i,Granulo[i]);
    fclose(fic);
  }*/

    delete[] bufferIn;
    return RES_OK;
  }

  template <class T1, class T2>
  RES_T ImFromSK_AreaForEachCC(const Image<T1> &imIn, int ScaleX, int ScaleY,
                               int ScaleZ, Image<T2> &imOut)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    int i, j, k, l, X, Y, Z, NbPixel, currentPixel, Ind;
    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int D = imIn.getDepth();

    UINT8 *bufferIn = new UINT8[W * H * D];
    if (bufferIn == 0) {
      // MORPHEE_REGISTER_ERROR("Error allocation bufferIn");
      return RES_ERR_BAD_ALLOCATION;
    }

    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    for (size_t i = 0; i < imIn.getPixelCount(); ++i)
      bufferIn[i] = ((pixelsIn[i] != 0) ? (1) : (0));

    std::queue<int> currentQueue, saveQueue;
    float Dist;
    for (i = W * H * D - 1; i >= 0; i--)
      if (bufferIn[i] == 1) {
        currentQueue.push(i);
        saveQueue.push(i);
        bufferIn[i] = 2;
        NbPixel     = 1;
        Dist        = 1;
        do {
          currentPixel = currentQueue.front();
          currentQueue.pop();
          X = currentPixel % W;
          Y = (currentPixel % (W * H) - X) / W;
          Z = (currentPixel - X - Y * W) / (W * H);

          // For all the neigbour
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      if (bufferIn[Ind] == 1) {
                        Dist += sqrt((float) ((j * ScaleX * j * ScaleX) +
                                              (k * ScaleY) * (k * ScaleY) +
                                              (l * ScaleZ) * (l * ScaleZ)));
                        NbPixel++;
                        bufferIn[Ind] = 2;
                        currentQueue.push(Ind);
                        saveQueue.push(Ind);
                      }
                    }
        } while (!currentQueue.empty());

        // if(NbPixel>=nbElt)NbPixel = nbElt-1;
        // Granulo[NbPixel]++;

        if (Dist > 255)
          j = 255;
        else
          j = (int) Dist;

        while (!saveQueue.empty()) {
          currentPixel = saveQueue.front();
          saveQueue.pop();
          pixelsOut[currentPixel] = j;
        }
      }

    delete[] bufferIn;
    return RES_OK;
  }

  template <class T>
  RES_T CountNbPixelOfNDG(const Image<T> &imIn, int NDG, int *NbPixel)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)

    *NbPixel = 0;

    typename Image<T>::lineType pixelsIn = imIn.getPixels();
    for (size_t i = 0; i < imIn.getPixelCount(); ++i)
      if (pixelsIn[i] == (ULONG) NDG)
        *NbPixel += 1;
    return RES_OK;
  }

  template <class T1>
  void ComputeMeanValue(T1 *imIn, int W, int H, int D, int *Somme,
                        int *NbComposanteConnexe)
  {
    int i, j, k, l, X, Y, Z, currentPixel, Ind;
    T1 NDG;

    std::queue<int> currentQueue;

    for (i = W * H * D - 1; i >= 0; i--)
      if (imIn[i] != 0) {
        NDG = imIn[i];

        *NbComposanteConnexe += 1;
        *Somme += NDG;

        currentQueue.push(i);
        imIn[i] = 0;

        do {
          currentPixel = currentQueue.front();
          currentQueue.pop();
          X = currentPixel % W;
          Y = (currentPixel % (W * H) - X) / W;
          Z = (currentPixel - X - Y * W) / (W * H);

          // For all the neigbour
          for (j = -1; j <= 1; j++)
            if (Z + j >= 0 && Z + j < D)
              for (k = -1; k <= 1; k++)
                if (X + k >= 0 && X + k < W)
                  for (l = -1; l <= 1; l++)
                    if (Y + l >= 0 && Y + l < H &&
                        (k != 0 || l != 0 || j != 0)) {
                      Ind = (X + k) + (Y + l) * W + (Z + j) * W * H;
                      if (imIn[Ind] == NDG) {
                        imIn[Ind] = 0;
                        currentQueue.push(Ind);
                      }
                    }
        } while (!currentQueue.empty());
      }
  }

  template <class T>
  RES_T MeanValueOf(const Image<T> &imIn, bool slideBySlide, double *Value)
  {
    // Check inputs
    ASSERT_ALLOCATED(&imIn)

    typename Image<T>::lineType pixelsIn = imIn.getPixels();

    int i, j, k;
    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int D = imIn.getDepth();

    T *bufferIn;

    if (slideBySlide)
      bufferIn = new T[W * H];
    else
      bufferIn = new T[W * H * D];

    if (bufferIn == 0) {
      // MORPHEE_REGISTER_ERROR("Memory error");
      return RES_ERR_BAD_ALLOCATION;
    }

    int Somme = 0, NbComposanteConnexe = 0;

    if (slideBySlide) {
      for (k = 0; k < D; k++) {
        for (j = 0; j < H; j++) // Copy
          for (i = 0; i < W; i++)
            bufferIn[i + j * W] = pixelsIn[i + j * W + k * W * H];

        ComputeMeanValue(bufferIn, W, H, 1, &Somme, &NbComposanteConnexe);
      }
    } else {
      for (k = W * H * D - 1; k >= 0; k--) // Copy
        bufferIn[k] = pixelsIn[k];

      ComputeMeanValue(bufferIn, W, H, D, &Somme, &NbComposanteConnexe);
    }

    *Value = Somme / (double) NbComposanteConnexe;

    delete[] bufferIn;

    return RES_OK;
  }

  template <class T1, class T2>
  RES_T GetConfusionMatrix(const Image<T1> &imMask, const Image<T2> &imIn2,
                           int seuil, int *FP, int *FN, int *TP, int *TN,
                           Image<RGB> &imOut, int WantImOut)
  {
    return RES_ERR_NOT_IMPLEMENTED;
    /*
    const T1 *imMaskPointer = imMask.rawPointer();
    const T2 *imInPointer = imIn2.rawPointer();
    T3 *imOutPointer;
    if(WantImOut==1)
      imOutPointer = imOut.rawPointer();

    int W = imMask.getWidth();
    int H = imMask.getHeight();

    *TN=0;
    *TP=0;
    *FN=0;
    *FP=0;

    for(int i=0;i<W*H;i++){
      if(imInPointer[i]<seuil){
        if(imMaskPointer[i]<seuil){
          *TN+=1;
          if(WantImOut==1){
            imOutPointer[i].channel1 = 0;
            imOutPointer[i].channel2 = 0;
            imOutPointer[i].channel3 = 0;
          }
        }
        else{
          *FN+=1;
          if(WantImOut==1){
            imOutPointer[i].channel1 = 200;
            imOutPointer[i].channel2 = 0;
            imOutPointer[i].channel3 = 0;
          }
        }
      }
      else{
        if(imMaskPointer[i]>=seuil){
          *TP+=1;
          if(WantImOut==1){
            imOutPointer[i].channel1 = 255;
            imOutPointer[i].channel2 = 255;
            imOutPointer[i].channel3 = 255;
          }
        }
        else{
          *FP+=1;
          if(WantImOut==1){
            imOutPointer[i].channel1 = 0;
            imOutPointer[i].channel2 = 200;
            imOutPointer[i].channel3 = 0;
          }
        }
      }
    }

    return RES_OK;
    */
  }

  // Distance Euclidienne
  template <class T1, class T2>
  RES_T ImLayerDist(Image<T1> &imIn, ULONG labelIn, ULONG labelOut, float dx,
                    float dy, float dz, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    ImageFreezer freeze(imOut);
    fill(imOut, T2(0));

    typename Image<T1>::lineType pixelsIn  = imIn.getPixels();
    typename Image<T2>::lineType pixelsOut = imOut.getPixels();

    int W = imIn.getWidth();
    int H = imIn.getHeight();
    int Z = imIn.getDepth();

    int nbJDE   = 0;
    int *posJDE = new int[5 * W * H];

    for (int k = 0; k < Z; k++) {
      for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++) {
          if (pixelsIn[i + j * W + k * W * H] == labelOut) {
            // For all the neighbour
            int touch = 0;
            for (int z = -1; z <= 1; z++) {
              if (k + z >= 0 && k + z < Z)
                for (int y = -1; y <= 1; y++)
                  if (j + y >= 0 && j + y < H)
                    for (int x = -1; x <= 1; x++)
                      if (i + x >= 0 && i + x < W &&
                          pixelsIn[(x + i) + (y + j) * W + (z + k) * W * H] ==
                              labelIn)
                        touch = 1;
            }
            if (touch) {
              posJDE[nbJDE] = i + j * W + k * W * H;
              nbJDE++;
            }
          }
        }
    }

    for (int k = 0; k < Z; k++) {
      for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++) {
          if (pixelsIn[i + j * W + k * W * H] == labelIn) {
            float dist, distMin = 9999999;
            for (int pix = 0; pix < nbJDE; pix++) {
              int x = posJDE[pix] % W;
              int y = (posJDE[pix] % (W * H) - x) / W;
              int z = (posJDE[pix] - x - y * W) / (W * H);

              dist = sqrt((i - x) * (i - x) * dx * dx +
                          (j - y) * (j - y) * dy * dy +
                          (k - z) * (k - z) * dz * dz);
              if (dist < distMin)
                distMin = dist;
            }
            pixelsOut[i + j * W + k * W * H] = distMin;
          }
        }
    }

    delete[] posJDE;
    return RES_OK;
  }

} // namespace smil
#endif
