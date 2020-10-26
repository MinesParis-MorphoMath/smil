#ifndef __FAST_LINE_HISTOGRAM_T_HPP__
#define __FAST_LINE_HISTOGRAM_T_HPP__

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/morpheeImage.hpp>

// Vincent Morard
// 21 octobre 2010

namespace morphee
{
  namespace FastLine
  {
    // BE CAREFUL : imIn and imOut must be two different images: it is not an
    // inPlace process!
    template <class T1, class T2>
    RES_C t_ImFastLineErodeHist_H(const Image<T1> &imIn, const int radius,
                                  Image<T2> &imOut)
    {
      // Check inputs
      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      int i, j, mini, W, H;
      const T1 *inLeft, *inRight, *aux, *bufferIn = imIn.rawPointer();
      T2 *current, *bufferOut = imOut.rawPointer();
      unsigned int Hist[256]; // Only for 8 bits images

      W = imIn.getWxSize();
      H = imIn.getWySize();

      // For all the line
      for (j = 0; j < H; j++) {
        mini = 255;
        memset(Hist, 0, 256 * sizeof(unsigned int));

        // Phase -1- : Start of the line
        current = &(bufferOut[j * W]);
        inLeft  = &(bufferIn[j * W]) - radius;
        inRight = &(bufferIn[j * W]) + radius;

        // on remplit l'histogramme
        for (aux = inRight - radius; aux < inRight; aux++) {
          Hist[*aux]++;
          if (*aux < mini)
            mini = *aux;
        }
        for (i = 0; i < radius; i++) {
          if (*inRight < mini)
            mini = *inRight;
          Hist[*inRight]++;
          // On cherche le minimum
          while (Hist[mini] <= 0) {
            mini++;
          }
          // Write the min and update the histogramme
          *current = mini;
          inLeft++;
          inRight++;
          current++;
        }

        // Fin de la première partie de l'image
        // Phase -2- Here we do not need to check if the pointer is inside the
        // boundaries
        for (i = radius; i < W - radius; i++) {
          if (*inRight < mini)
            mini = *inRight;
          Hist[*inRight]++;

          // On cherche le minimum
          while (Hist[mini] <= 0) {
            mini++;
          }

          // Write the min and update the histogramme
          *current = mini;
          Hist[*inLeft]--;
          inLeft++;
          inRight++;
          current++;
        }

        // Phase -3- : End of the line.
        for (i = W - radius; i < W; i++) {
          // On cherche le minimum
          while (Hist[mini] <= 0) {
            mini++;
          }
          // Write the min and update the histogramme
          *current = mini;
          Hist[*inLeft]--;
          inLeft++;
          inRight++;
          current++;
        }

        // End of the line
      }

      return RES_OK;
    }

    // BE CAREFUL : imIn and imOut must be two different images: it is not an
    // inPlace process!
    template <class T1, class T2>
    RES_C t_ImFastLineDilateHist_H(const Image<T1> &imIn, const int radius,
                                   Image<T2> &imOut)
    {
      // Check inputs
      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      int i, j, maxi, W, H;
      const T1 *inLeft, *inRight, *aux, *bufferIn = imIn.rawPointer();
      T2 *current, *bufferOut = imOut.rawPointer();
      unsigned int Hist[256]; // Only for 8 bits images

      W = imIn.getWxSize();
      H = imIn.getWySize();

      // For all the line
      for (j = 0; j < H; j++) {
        maxi = 0;
        memset(Hist, 0, 256 * sizeof(unsigned int));

        // Phase -1- : Start of the line
        current = &(bufferOut[j * W]);
        inLeft  = &(bufferIn[j * W]) - radius;
        inRight = &(bufferIn[j * W]) + radius;

        // on remplit l'histogramme
        for (aux = inRight - radius; aux < inRight; aux++) {
          Hist[*aux]++;
          if (*aux > maxi)
            maxi = *aux;
        }
        for (i = 0; i < radius; i++) {
          if (*inRight > maxi)
            maxi = *inRight;
          Hist[*inRight]++;
          // On cherche le minimum
          while (Hist[maxi] <= 0) {
            maxi--;
          }
          // Write the min and update the histogramme
          *current = maxi;
          inLeft++;
          inRight++;
          current++;
        }

        // Fin de la première partie de l'image
        // Phase -2- Here we do not need to check if the pointer is inside the
        // boundaries
        for (i = radius; i < W - radius; i++) {
          if (*inRight > maxi)
            maxi = *inRight;
          Hist[*inRight]++;

          // On cherche le minimum
          while (Hist[maxi] <= 0) {
            maxi--;
          }

          // Write the min and update the histogramme
          *current = maxi;
          Hist[*inLeft]--;
          inLeft++;
          inRight++;
          current++;
        }

        // Phase -3- : End of the line.
        for (i = W - radius; i < W; i++) {
          // On cherche le minimum
          while (Hist[maxi] <= 0) {
            maxi--;
          }
          // Write the min and update the histogramme
          *current = maxi;
          Hist[*inLeft]--;
          inLeft++;
          inRight++;
          current++;
        }

        // End of the line
      }

      return RES_OK;
    }

    // BE CAREFUL : imIn and imOut must be two different images: it is not an
    // inPlace process!
    template <class T1, class T2>
    RES_C t_ImFastLineErodeHist_V(const Image<T1> &imIn, const int radius,
                                  Image<T2> &imOut)
    {
      // Check inputs
      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      int i, j, mini, W, H;
      const T1 *inLeft, *inRight, *aux, *bufferIn = imIn.rawPointer();
      T2 *current, *bufferOut = imOut.rawPointer();
      unsigned int Hist[256]; // Only for 8 bits images

      W = imIn.getWxSize();
      H = imIn.getWySize();

      // For all the line
      for (i = 0; i < W; i++) {
        mini = 255;
        memset(Hist, 0, 256 * sizeof(unsigned int));

        // Phase -1- : Start of the line
        current = &(bufferOut[i]);
        inLeft  = &(bufferIn[i]) - (radius * W);
        inRight = &(bufferIn[i]) + (radius * W);

        // on remplit l'histogramme
        for (aux = inRight - (radius * W); aux < inRight; aux += W) {
          Hist[*aux]++;
          if (*aux < mini)
            mini = *aux;
        }
        for (j = 0; j < radius; j++) {
          if (*inRight < mini)
            mini = *inRight;
          Hist[*inRight]++;
          // On cherche le minimum
          while (Hist[mini] <= 0) {
            mini++;
          }
          // Write the min and update the histogramme
          *current = mini;
          inLeft += W;
          inRight += W;
          current += W;
        }

        // Fin de la première partie de l'image
        // Phase -2- Here we do not need to check if the pointer is inside the
        // boundaries
        for (j = radius; j < H - radius; j++) {
          if (*inRight < mini)
            mini = *inRight;
          Hist[*inRight]++;

          // On cherche le minimum
          while (Hist[mini] <= 0) {
            mini++;
          }

          // Write the min and update the histogramme
          *current = mini;
          Hist[*inLeft]--;
          inLeft += W;
          inRight += W;
          current += W;
        }

        // Phase -3- : End of the line.
        for (j = H - radius; j < H; j++) {
          // On cherche le minimum
          while (Hist[mini] <= 0) {
            mini++;
          }
          // Write the min and update the histogramme
          *current = mini;
          Hist[*inLeft]--;
          inLeft += W;
          inRight += W;
          current += W;
        }

        // End of the line
      }

      return RES_OK;
    }

    // BE CAREFUL : imIn and imOut must be two different images: it is not an
    // inPlace process!
    template <class T1, class T2>
    RES_C t_ImFastLineDilateHist_V(const Image<T1> &imIn, const int radius,
                                   Image<T2> &imOut)
    {
      // Check inputs
      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      int i, j, maxi, W, H;
      const T1 *inLeft, *inRight, *aux, *bufferIn = imIn.rawPointer();
      T2 *current, *bufferOut = imOut.rawPointer();
      unsigned int Hist[256]; // Only for 8 bits images

      W = imIn.getWxSize();
      H = imIn.getWySize();

      // For all the line
      for (i = 0; i < W; i++) {
        maxi = 0;
        memset(Hist, 0, 256 * sizeof(unsigned int));

        // Phase -1- : Start of the line
        current = &(bufferOut[i]);
        inLeft  = &(bufferIn[i]) - (radius * W);
        inRight = &(bufferIn[i]) + (radius * W);

        // on remplit l'histogramme
        for (aux = inRight - (radius * W); aux < inRight; aux += W) {
          Hist[*aux]++;
          if (*aux > maxi)
            maxi = *aux;
        }
        for (j = 0; j < radius; j++) {
          if (*inRight > maxi)
            maxi = *inRight;
          Hist[*inRight]++;
          // On cherche le minimum
          while (Hist[maxi] <= 0) {
            maxi--;
          }
          // Write the min and update the histogramme
          *current = maxi;
          inLeft += W;
          inRight += W;
          current += W;
        }

        // Fin de la première partie de l'image
        // Phase -2- Here we do not need to check if the pointer is inside the
        // boundaries
        for (j = radius; j < H - radius; j++) {
          if (*inRight > maxi)
            maxi = *inRight;
          Hist[*inRight]++;

          // On cherche le minimum
          while (Hist[maxi] <= 0) {
            maxi--;
          }

          // Write the min and update the histogramme
          *current = maxi;
          Hist[*inLeft]--;
          inLeft += W;
          inRight += W;
          current += W;
        }

        // Phase -3- : End of the line.
        for (j = H - radius; j < H; j++) {
          // On cherche le minimum
          while (Hist[maxi] <= 0) {
            maxi--;
          }
          // Write the min and update the histogramme
          *current = maxi;
          Hist[*inLeft]--;
          inLeft += W;
          inRight += W;
          current += W;
        }

        // End of the line
      }

      return RES_OK;
    }

    //
    // INTERFACE
    //

    // BE CAREFUL : imIn and imOut must be two different images: it is not an
    // inPlace process!
    template <class T1, class T2>
    RES_C t_ImFastLineDilate_Hist(const Image<T1> &imIn, const int radius,
                                  bool horizontal, Image<T2> &imOut)
    {
      if (horizontal)
        return t_ImFastLineDilateHist_H(imIn, radius, imOut);
      return t_ImFastLineDilateHist_V(imIn, radius, imOut);
    }

    // BE CAREFUL : imIn and imOut must be two different images: it is not an
    // inPlace process!
    template <class T1, class T2>
    RES_C t_ImFastLineErode_Hist(const Image<T1> &imIn, const int radius,
                                 bool horizontal, Image<T2> &imOut)
    {
      if (horizontal)
        return t_ImFastLineErodeHist_H(imIn, radius, imOut);
      return t_ImFastLineErodeHist_V(imIn, radius, imOut);
    }

  } // namespace FastLine
} // namespace morphee

#endif
