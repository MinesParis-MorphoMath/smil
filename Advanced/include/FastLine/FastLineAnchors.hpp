#ifndef __FAST_LINE_ANCHORS_T_HPP__
#define __FAST_LINE_ANCHORS_T_HPP__

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/morpheeImage.hpp>

#if (WITH_ANCHORS_FUNCTION)

// Vincent Morard
// 12 octobre : Interface to Morph-M of the opening via Anchors

// http://www2.ulg.ac.be/telecom/research/libmorpho.html
// libmorpho
// Authors:
// Renaud Dardenne
// Marc Van Droogenbroeck

/* LIBMORPHO
 *
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU  General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 * You should have received a copy of the GNU  General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

namespace morphee
{
  namespace FastLine
  {
    /*!
    * \fn int openingByAnchor_1D_horizontal(uint8_t *imageIn, uint8_t *imageOut,
    int imageWidth, int imageHeight, int size)
    * \param[in]  *imageIn Input buffer
    * \param[out]  *imageOut Output buffer
    * \param[in]  imageWidth Width of the image buffer
    * \param[in]  imageHeight Height of the image buffer
    * \param[in]  size (= width in tems of pixels) of the linear structuring
    element
    * \return Returns MORPHO_SUCCESS upon success, MORPHO_ERROR otherwise.
    *
    * \brief Opening with an horizontal linear segment
    *
    * \ingroup libmorpho
    *
    * Opening with an horizontal linear segment whose size is given in pixels.
    * For full technical details please refer to \ref detailsPage
    * or to
    * - M. Van Droogenbroeck and M. Buckley. <b>Morphological erosions and
    openings: fast algorithms based on anchors</b>. <em>Journal of Mathematical
    Imaging and Vision</em>, Special Issue on Mathematical Morphology after 40
    Years, 22(2-3):121-142, May 2005.
    *
    * \author      Marc Van Droogenbroeck
    */

    template <class T1, class T2>
    RES_C openingByAnchor_1D_horizontal(T1 *imageIn, T2 *imageOut,
                                        int imageWidth, int imageHeight,
                                        int size)
    {
      T1 *in;
      T2 *out, *aux, *end;
      T2 *outLeft, *outRight, *current, *sentinel;
      T2 min;
      int j, imageWidthMinus1, sizeMinus1;
      int *histo, nbrBytes;

      in               = imageIn;
      out              = imageOut;
      imageWidthMinus1 = imageWidth - 1;
      sizeMinus1       = size - 1;

      /* Copy the input into the output */
      memcpy(out, in, imageWidth * imageHeight * sizeof(UINT8));

      /* Initialisation of the histogram */
      nbrBytes = 256 * sizeof(int);
      histo    = (int *) malloc(nbrBytes);

      /* Computation */
      out = imageOut;

      /* Row by row */
      for (j = 0; j < imageHeight; j++) {
        /* Initialisation of both extremities of a line */
        outLeft  = out + (j * imageWidth);
        outRight = outLeft + imageWidthMinus1;

        /* Handling of both sides */
        /* Left side */
        while ((outLeft < outRight) && (*outLeft >= *(outLeft + 1))) {
          outLeft++;
        }

        /* Right side */
        while ((outLeft < outRight) && (*(outRight - 1) <= *outRight)) {
          outRight--;
        }

        /* Enters in the loop */
      startLine:
        min     = *outLeft;
        current = outLeft + 1;
        while ((current < outRight) && (*current <= min)) {
          min = *current;
          outLeft++;
          current++;
        }
        sentinel = outLeft + size;
        if (sentinel > outRight) {
          goto finishLine;
        }

        /* We ran "size" pixels ahead */
        current++;
        while (current < sentinel) {
          if (*current <= min) /* We have found a new minimum */
          {
            end = current;
            outLeft++;
            while (outLeft < end) {
              *outLeft = min;
              outLeft++;
            }
            outLeft = current;
            goto startLine;
          }
          current++;
        }

        /* We did not find a smaller value in the segment in reach
         * of outLeft; current is the first position outside the
         * reach of outLeft
         */
        if (*current <= min) {
          end = current;
          outLeft++;
          while (outLeft < end) {
            *outLeft = min;
            outLeft++;
          }
          outLeft = current;
          goto startLine;
        } else /* We can not avoid computing the histogram */
        {
          memset(histo, 0, nbrBytes);
          outLeft++;
          for (aux = outLeft; aux <= current; aux++) {
            histo[*aux]++;
          }
          min++;
          while (histo[min] <= 0) {
            min++;
          }
          histo[*outLeft]--;
          *outLeft = min;
          histo[min]++;
        }

        /* We just follow the pixels, update the histogram and look for
         * the minimum */
        while (current < outRight) {
          current++;
          if (*current <= min) {
            /* We have found a new mimum */
            end = current;
            outLeft++;
            while (outLeft < end) {
              *outLeft = min;
              outLeft++;
            }
            outLeft = current;
            goto startLine;
          } else {
            /* Update the histogram */
            histo[*current]++;
            histo[*outLeft]--;
            /* Recompute the minimum */
            while (histo[min] <= 0) {
              min++;
            }
            outLeft++;
            histo[*outLeft]--;
            *outLeft = min;
            histo[min]++;
          }
        }

        /* We have to finish the line */
        while (outLeft < outRight) {
          histo[*outLeft]--;
          while (histo[min] <= 0) {
            min++;
          }
          outLeft++;
          histo[*outLeft]--;
          *outLeft = min;
          histo[min]++;
        }

      finishLine:
        while (outLeft < outRight) {
          if (*outLeft <= *outRight) {
            min = *outRight;
            outRight--;
            if (*outRight > min) {
              *outRight = min;
            }
          } else {
            min = *outLeft;
            outLeft++;
            if (*outLeft > min) {
              *outLeft = min;
            }
          }
        }
      }

      /* Free memory */
      free(histo);
      return RES_OK;
    }

    /*!
    * \fn int openingByAnchor_1D_vertical(uint8_t *imageIn, uint8_t *imageOut,
    int imageWidth, int imageHeight, int size)
    * \param[in]  *imageIn Input buffer
    * \param[out]  *imageOut Output buffer
    * \param[in]  imageWidth Width of the image buffer
    * \param[in]  imageHeight Height of the image buffer
    * \param[in]  size (= width in tems of pixels) of the linear structuring
    element
    * \return Returns MORPHO_SUCCESS upon success, MORPHO_ERROR otherwise.
    *
    * \brief Opening with a vertical linear segment
    *
    * \ingroup libmorpho
    *
    * Opening with a vertical linear segment whose size is given in pixels.
    * For full technical details please refer to \ref detailsPage
    * or to
    * - M. Van Droogenbroeck and M. Buckley. <b>Morphological erosions and
    openings: fast algorithms based on anchors</b>. <em>Journal of Mathematical
    Imaging and Vision</em>, Special Issue on Mathematical Morphology after 40
    Years, 22(2-3):121-142, May 2005.
    *
    * \author      Marc Van Droogenbroeck
    */
    RES_C openingByAnchor_1D_vertical(const UINT8 *imageIn, UINT8 *imageOut,
                                      int imageWidth, int imageHeight, int size)
    {
      const UINT8 *in;
      UINT8 *out, *aux, *end;
      UINT8 *outUp, *outDown, *current, *sentinel;
      UINT8 min;
      int j, imageJump, sizeJump, sizeMinus1;
      int *histo, nbrBytes;

      in         = imageIn;
      out        = imageOut;
      imageJump  = imageWidth * (imageHeight - 1);
      sizeMinus1 = size - 1;
      sizeJump   = size * imageWidth;

      /* Copy the input into the output */
      memcpy(out, in, imageWidth * imageHeight * sizeof(UINT8));

      /* Initialisation of the histogram */
      nbrBytes = 256 * sizeof(int);
      histo    = (int *) malloc(nbrBytes);

      /* Computation */
      out = imageOut;

      /* Row by row */
      for (j = 0; j < imageWidth; j++) {
        /* Initialisation of both extremities of a line */
        outUp   = out + j;
        outDown = outUp + imageJump;

        /* Handling of both sides */
        /* Left side */
        while ((outUp < outDown) && (*outUp >= *(outUp + imageWidth))) {
          outUp += imageWidth;
        }

        /* Right side */
        while ((outUp < outDown) && (*(outDown - imageWidth) <= *outDown)) {
          outDown -= imageWidth;
        }

        /* Enters in the loop */
      startLine:
        min     = *outUp;
        current = outUp + imageWidth;
        while ((current < outDown) && (*current <= min)) {
          min = *current;
          outUp += imageWidth;
          current += imageWidth;
        }
        sentinel = outUp + sizeJump;
        if (sentinel > outDown) {
          goto finishLine;
        }

        /* We ran "size" pixels ahead */
        current += imageWidth;
        while (current < sentinel) {
          if (*current <= min) /* We have found a new minimum */
          {
            end = current;
            outUp += imageWidth;
            while (outUp < end) {
              *outUp = min;
              outUp += imageWidth;
            }
            outUp = current;
            goto startLine;
          }
          current += imageWidth;
        }

        /* We did not find a smaller value in the segment in reach
         * of outUp; current is the first position outside the
         * reach of outUp
         */
        if (*current <= min) {
          end = current;
          outUp += imageWidth;
          while (outUp < end) {
            *outUp = min;
            outUp += imageWidth;
          }
          outUp = current;
          goto startLine;
        } else /* We can not avoid computing the histogram */
        {
          memset(histo, 0, nbrBytes);
          outUp += imageWidth;
          for (aux = outUp; aux <= current; aux += imageWidth) {
            histo[*aux]++;
          }
          min++;
          while (histo[min] <= 0) {
            min++;
          }
          histo[*outUp]--;
          *outUp = min;
          histo[min]++;
        }

        /* We just follow the pixels, update the histogram and look for
         * the minimum */
        while (current < outDown) {
          current += imageWidth;
          if (*current <= min) {
            /* We have found a new mimum */
            end = current;
            outUp += imageWidth;
            while (outUp < end) {
              *outUp = min;
              outUp += imageWidth;
            }
            outUp = current;
            goto startLine;
          } else {
            /* Update the histogram */
            histo[*current]++;
            histo[*outUp]--;
            /* Recompute the minimum */
            while (histo[min] <= 0) {
              min++;
            }
            outUp += imageWidth;
            histo[*outUp]--;
            *outUp = min;
            histo[min]++;
          }
        }

        /* We have to finish the line */
        while (outUp < outDown) {
          histo[*outUp]--;
          while (histo[min] <= 0) {
            min++;
          }
          outUp += imageWidth;
          histo[*outUp]--;
          *outUp = min;
          histo[min]++;
        }

      finishLine:
        while (outUp < outDown) {
          if (*outUp <= *outDown) {
            min = *outDown;
            outDown -= imageWidth;
            if (*outDown > min) {
              *outDown = min;
            }
          } else {
            min = *outUp;
            outUp += imageWidth;
            if (*outUp > min) {
              *outUp = min;
            }
          }
        }
      }

      /* Free memory */
      free(histo);

      return RES_OK;
    }

    /*!
    * \fn int closingByAnchor_1D_horizontal(uint8_t *imageIn, uint8_t *imageOut,
    int imageWidth, int imageHeight, int size)
    * \param[in]  *imageIn Input buffer
    * \param[out]  *imageOut Output buffer
    * \param[in]  imageWidth Width of the image buffer
    * \param[in]  imageHeight Height of the image buffer
    * \param[in]  size (= width in tems of pixels) of the linear structuring
    element
    * \return Returns MORPHO_SUCCESS upon success, MORPHO_ERROR otherwise.
    *
    * \brief Closing with an horizontal linear segment
    *
    * \ingroup libmorpho
    *
    * Closing with an horizontal linear segment whose size is given in pixels.
    * For full technical details please refer to \ref detailsPage
    * or to
    * - M. Van Droogenbroeck and M. Buckley. <b>Morphological erosions and
    openings: fast algorithms based on anchors</b>. <em>Journal of Mathematical
    Imaging and Vision</em>, Special Issue on Mathematical Morphology after 40
    Years, 22(2-3):121-142, May 2005.
    *
    * \author      Marc Van Droogenbroeck
    */
    RES_C closingByAnchor_1D_horizontal(const UINT8 *imageIn, UINT8 *imageOut,
                                        int imageWidth, int imageHeight,
                                        int size)
    {
      const UINT8 *in;
      UINT8 *out, *aux, *end;
      UINT8 *outLeft, *outRight, *current, *sentinel;
      UINT8 max;
      int j, imageWidthMinus1, sizeMinus1;
      int *histo, nbrBytes;

      in               = imageIn;
      out              = imageOut;
      imageWidthMinus1 = imageWidth - 1;
      sizeMinus1       = size - 1;

      /* Copy the input into the output */
      memcpy(out, in, imageWidth * imageHeight * sizeof(UINT8));

      /* Initialisation of the histogram */
      nbrBytes = 256 * sizeof(int);
      histo    = (int *) malloc(nbrBytes);

      /* Computation */
      out = imageOut;

      /* Row by row */
      for (j = 0; j < imageHeight; j++) {
        /* Initialisation of both extremities of a line */
        outLeft  = out + (j * imageWidth);
        outRight = outLeft + imageWidthMinus1;

        /* Handling of both sides */
        /* Left side */
        while ((outLeft < outRight) && (*outLeft <= *(outLeft + 1))) {
          outLeft++;
        }

        /* Right side */
        while ((outLeft < outRight) && (*(outRight - 1) >= *outRight)) {
          outRight--;
        }

        /* Enters in the loop */
      startLine:
        max     = *outLeft;
        current = outLeft + 1;
        while ((current < outRight) && (*current >= max)) {
          max = *current;
          outLeft++;
          current++;
        }
        sentinel = outLeft + size;
        if (sentinel > outRight) {
          goto finishLine;
        }

        /* We ran "size" pixels ahead */
        current++;
        while (current < sentinel) {
          if (*current >= max) /* We have found a new maximum */
          {
            end = current;
            outLeft++;
            while (outLeft < end) {
              *outLeft = max;
              outLeft++;
            }
            outLeft = current;
            goto startLine;
          }
          current++;
        }

        /* We did not find a larger value in the segment in reach
         * of outLeft; current is the first position outside the
         * reach of outLeft
         */
        if (*current >= max) {
          end = current;
          outLeft++;
          while (outLeft < end) {
            *outLeft = max;
            outLeft++;
          }
          outLeft = current;
          goto startLine;
        } else /* We can not avoid computing the histogram */
        {
          memset(histo, 0, nbrBytes);
          outLeft++;
          for (aux = outLeft; aux <= current; aux++) {
            histo[*aux]++;
          }
          max--;
          while (histo[max] <= 0) {
            max--;
          }
          histo[*outLeft]--;
          *outLeft = max;
          histo[max]++;
        }

        /* We just follow the pixels, update the histogram and look for
         * the maximum */
        while (current < outRight) {
          current++;
          if (*current >= max) {
            /* We have found a new maximum */
            end = current;
            outLeft++;
            while (outLeft < end) {
              *outLeft = max;
              outLeft++;
            }
            outLeft = current;
            goto startLine;
          } else {
            /* Update the histogram */
            histo[*current]++;
            histo[*outLeft]--;
            /* Recompute the minimum */
            while (histo[max] <= 0) {
              max--;
            }
            outLeft++;
            histo[*outLeft]--;
            *outLeft = max;
            histo[max]++;
          }
        }

        /* We have to finish the line */
        while (outLeft < outRight) {
          histo[*outLeft]--;
          while (histo[max] <= 0) {
            max--;
          }
          outLeft++;
          histo[*outLeft]--;
          *outLeft = max;
          histo[max]++;
        }

      finishLine:
        while (outLeft < outRight) {
          if (*outLeft <= *outRight) {
            max = *outRight;
            outRight--;
            if (*outRight < max) {
              *outRight = max;
            }
          } else {
            max = *outLeft;
            outLeft++;
            if (*outLeft < max) {
              *outLeft = max;
            }
          }
        }
      }

      /* Free memory */
      free(histo);

      return RES_OK;
    }

    /*!
    * \fn int closingByAnchor_1D_vertical(uint8_t *imageIn, uint8_t *imageOut,
    int imageWidth, int imageHeight, int size)
    * \param[in]  *imageIn Input buffer
    * \param[out]  *imageOut Output buffer
    * \param[in]  imageWidth Width of the image buffer
    * \param[in]  imageHeight Height of the image buffer
    * \param[in]  size (= width in tems of pixels) of the linear structuring
    element
    * \return Returns MORPHO_SUCCESS upon success, MORPHO_ERROR otherwise.
    *
    * \brief Closing with a vertical linear segment
    *
    * \ingroup libmorpho
    *
    * Closing with a vertical linear segment whose size is given in pixels.
    * For full technical details please refer to \ref detailsPage
    * or to
    * - M. Van Droogenbroeck and M. Buckley. <b>Morphological erosions and
    openings: fast algorithms based on anchors</b>. <em>Journal of Mathematical
    Imaging and Vision</em>, Special Issue on Mathematical Morphology after 40
    Years, 22(2-3):121-142, May 2005.
    *
    * \author      Marc Van Droogenbroeck
    */
    RES_C closingByAnchor_1D_vertical(const UINT8 *imageIn, UINT8 *imageOut,
                                      int imageWidth, int imageHeight, int size)
    {
      const UINT8 *in;
      UINT8 *out, *aux, *end;
      UINT8 *outUp, *outDown, *current, *sentinel;
      UINT8 max;
      int j, imageJump, sizeJump, sizeMinus1;
      int *histo, nbrBytes;

      in         = imageIn;
      out        = imageOut;
      imageJump  = imageWidth * (imageHeight - 1);
      sizeMinus1 = size - 1;
      sizeJump   = size * imageWidth;

      /* Copy the input into the output */
      memcpy(out, in, imageWidth * imageHeight * sizeof(UINT8));

      /* Initialisation of the histogram */
      nbrBytes = 256 * sizeof(int);
      histo    = (int *) malloc(nbrBytes);

      /* Computation */
      out = imageOut;

      /* Row by row */
      for (j = 0; j < imageWidth; j++) {
        /* Initialisation of both extremities of a line */
        outUp   = out + j;
        outDown = outUp + imageJump;

        /* Handling of both sides */
        /* Up side */
        while ((outUp < outDown) && (*outUp <= *(outUp + imageWidth))) {
          outUp += imageWidth;
        }

        /* Down side */
        while ((outUp < outDown) && (*(outDown - imageWidth) >= *outDown)) {
          outDown -= imageWidth;
        }

        /* Enters in the loop */
      startLine:
        max     = *outUp;
        current = outUp + imageWidth;
        while ((current < outDown) && (*current >= max)) {
          max = *current;
          outUp += imageWidth;
          current += imageWidth;
        }
        sentinel = outUp + sizeJump;
        if (sentinel > outDown) {
          goto finishLine;
        }

        /* We ran "size" pixels ahead */
        current += imageWidth;
        while (current < sentinel) {
          if (*current >= max) /* We have found a new maximum */
          {
            end = current;
            outUp += imageWidth;
            while (outUp < end) {
              *outUp = max;
              outUp += imageWidth;
            }
            outUp = current;
            goto startLine;
          }
          current += imageWidth;
        }

        /* We did not found a larger value in the segment in reach
         * of outUp; current is the first position outside the
         * reach of outUp
         */
        if (*current >= max) {
          end = current;
          outUp += imageWidth;
          while (outUp < end) {
            *outUp = max;
            outUp += imageWidth;
          }
          outUp = current;
          goto startLine;
        } else /* We can not avoid computing the histogram */
        {
          memset(histo, 0, nbrBytes);
          outUp += imageWidth;
          for (aux = outUp; aux <= current; aux += imageWidth) {
            histo[*aux]++;
          }
          max--;
          while (histo[max] <= 0) {
            max--;
          }
          histo[*outUp]--;
          *outUp = max;
          histo[max]++;
        }

        /* We just follow the pixels, update the histogram and look for
         * the maximum */
        while (current < outDown) {
          current += imageWidth;
          if (*current >= max) {
            /* We have found a new maximum */
            end = current;
            outUp += imageWidth;
            while (outUp < end) {
              *outUp = max;
              outUp += imageWidth;
            }
            outUp = current;
            goto startLine;
          } else {
            /* Update the histogram */
            histo[*current]++;
            histo[*outUp]--;
            /* Recompute the minimum */
            while (histo[max] <= 0) {
              max--;
            }
            outUp += imageWidth;
            histo[*outUp]--;
            *outUp = max;
            histo[max]++;
          }
        }

        /* We have to finish the line */
        while (outUp < outDown) {
          histo[*outUp]--;
          while (histo[max] <= 0) {
            max--;
          }
          outUp += imageWidth;
          histo[*outUp]--;
          *outUp = max;
          histo[max]++;
        }

      finishLine:
        while (outUp < outDown) {
          if (*outUp <= *outDown) {
            max = *outDown;
            outDown -= imageWidth;
            if (*outDown < max) {
              *outDown = max;
            }
          } else {
            max = *outUp;
            outUp += imageWidth;
            if (*outUp < max) {
              *outUp = max;
            }
          }
        }
      }

      /* Free memory */
      free(histo);

      return RES_OK;
    }

    /*!
    * \fn int erosionByAnchor_1D_horizontal(uint8_t *imageIn, uint8_t *imageOut,
    int imageWidth, int imageHeight, int size)
    * \param[in]  *imageIn Input buffer
    * \param[out]  *imageOut Output buffer
    * \param[in]  imageWidth Width of the image buffer
    * \param[in]  imageHeight Height of the image buffer
    * \param[in]  size (= width in tems of pixels) of the linear structuring
    element
    * \return Returns MORPHO_SUCCESS upon success, MORPHO_ERROR otherwise.
    *
    * \brief Erosion with an horizontal linear segment
    *
    * \ingroup libmorpho
    *
    * Erosion with an horizontal linear segment whose size is given in pixels.
    * For full technical details please refer to \ref detailsPage
    * or to
    * - M. Van Droogenbroeck and M. Buckley. <b>Morphological erosions and
    openings: fast algorithms based on anchors</b>. <em>Journal of Mathematical
    Imaging and Vision</em>, Special Issue on Mathematical Morphology after 40
    Years, 22(2-3):121-142, May 2005.
    *
    * \author      Marc Van Droogenbroeck
    */
    RES_C erosionByAnchor_1D_horizontal(const UINT8 *imageIn, UINT8 *imageOut,
                                        int imageWidth, int imageHeight,
                                        int size)
    {
      const UINT8 *in, *inLeft, *inRight, *current, *sentinel, *aux;
      UINT8 *out;
      UINT8 *outLeft, *outRight;
      UINT8 min;
      int i, j, imageWidthMinus1, sizeMinus1;
      int *histo, nbrBytes;
      int middle;

      in               = (UINT8 *) imageIn;
      out              = (UINT8 *) imageOut;
      imageWidthMinus1 = imageWidth - 1;
      sizeMinus1       = size - 1;
      middle           = size / 2;

      /* Initialisation of the histogram */
      nbrBytes = 256 * sizeof(int);
      histo    = (int *) malloc(nbrBytes);

      /* Computation */
      /* Row by row */
      for (j = 0; j < imageHeight; j++) {
        /* Initialisation of both extremities of a line */
        inLeft   = in + (j * imageWidth);
        outLeft  = out + (j * imageWidth);
        inRight  = inLeft + imageWidthMinus1;
        outRight = outLeft + imageWidthMinus1;

        /* Handles the left border */
        /* First half of the structuring element */
        memset(histo, 0, nbrBytes);
        min = *inLeft;
        histo[min]++;
        for (i = 0; i < middle; i++) {
          inLeft++;
          histo[*inLeft]++;
          if (*inLeft < min) {
            min = *inLeft;
          }
        }
        *outLeft = min;

        /* Second half of the structuring element */
        for (i = 0; i < size - middle - 1; i++) {
          inLeft++;
          outLeft++;
          histo[*inLeft]++;
          if (*inLeft < min) {
            min = *inLeft;
          }
          *outLeft = min;
        }

        /* Use the histogram as long as we have not found a new minimum */
        while ((inLeft < inRight) && (min <= *(inLeft + 1))) {
          inLeft++;
          outLeft++;
          histo[*(inLeft - size)]--;
          histo[*inLeft]++;
          while (histo[min] <= 0) {
            min++;
          }
          *outLeft = min;
        }

        /* Enters in the loop */
        min = *outLeft;

      startLine:
        current = inLeft + 1;
        while ((current < inRight) && (*current <= min)) {
          min = *current;
          outLeft++;
          *outLeft = min;
          current++;
        }
        inLeft   = current - 1;
        sentinel = inLeft + size;
        if (sentinel > inRight) {
          goto finishLine;
        }
        outLeft++;
        *outLeft = min;

        /* We ran "size" pixels ahead */
        current++;
        while (current < sentinel) {
          if (*current <= min) /* We have found a new minimum */
          {
            min = *current;
            outLeft++;
            *outLeft = min;
            inLeft   = current;
            goto startLine;
          }
          current++;
          outLeft++;
          *outLeft = min;
        }

        /* We did not find a smaller value in the segment in reach
         * of inLeft; current is the first position outside the
         * reach of inLeft
         */
        if (*current <= min) {
          min = *current;
          outLeft++;
          *outLeft = min;
          inLeft   = current;
          goto startLine;
        } else /* We can not avoid computing the histogram */
        {
          memset(histo, 0, nbrBytes);
          inLeft++;
          outLeft++;
          for (aux = inLeft; aux <= current; aux++) {
            histo[*aux]++;
          }
          min++;
          while (histo[min] <= 0) {
            min++;
          }
          *outLeft = min;
        }

        /* We just follow the pixels, update the histogram and look for
         * the minimum */
        while (current < inRight) {
          current++;
          if (*current <= min) {
            /* We have found a new mimum */
            min = *current;
            outLeft++;
            *outLeft = min;
            inLeft   = current;
            goto startLine;
          } else {
            /* Update the histogram */
            histo[*current]++;
            histo[*inLeft]--;
            /* Recompute the minimum */
            while (histo[min] <= 0) {
              min++;
            }
            inLeft++;
            outLeft++;
            *outLeft = min;
          }
        }

      finishLine:
        /* Handles the right border */
        /* First half of the structuring element */
        memset(histo, 0, nbrBytes);
        min = *inRight;
        histo[min]++;
        for (i = 0; i < middle; i++) {
          inRight--;
          histo[*inRight]++;
          if (*inRight < min) {
            min = *inRight;
          }
        }
        *outRight = min;

        /* Second half of the structuring element */
        for (i = 0; (i < size - middle - 1) && (outLeft < outRight); i++) {
          inRight--;
          outRight--;
          histo[*inRight]++;
          if (*inRight < min) {
            min = *inRight;
          }
          *outRight = min;
        }

        /* Use the histogram as long as we have not found a new minimum */
        while (outLeft < outRight) {
          inRight--;
          outRight--;
          histo[*(inRight + size)]--;
          histo[*inRight]++;
          if (*inRight < min) {
            min = *inRight;
          }
          while (histo[min] <= 0) {
            min++;
          }
          *outRight = min;
        }
      }

      /* Free memory */
      free(histo);

      return RES_OK;
    }

    /*!
    * \fn int erosionByAnchor_1D_vertical(uint8_t *imageIn, uint8_t *imageOut,
    int imageWidth, int imageHeight, int size)
    * \param[in]  *imageIn Input buffer
    * \param[out]  *imageOut Output buffer
    * \param[in]  imageWidth Width of the image buffer
    * \param[in]  imageHeight Height of the image buffer
    * \param[in]  size (= width in tems of pixels) of the linear structuring
    element
    * \return Returns MORPHO_SUCCESS upon success, MORPHO_ERROR otherwise.
    *
    * \brief Erosion with a vertical linear segment
    *
    * \ingroup libmorpho
    *
    * Erosion with a vertical linear segment whose size is given in pixels.
    * For full technical details please refer to \ref detailsPage
    * or to
    * - M. Van Droogenbroeck and M. Buckley. <b>Morphological erosions and
    openings: fast algorithms based on anchors</b>. <em>Journal of Mathematical
    Imaging and Vision</em>, Special Issue on Mathematical Morphology after 40
    Years, 22(2-3):121-142, May 2005.
    *
    * \author      Marc Van Droogenbroeck
    */
    RES_C erosionByAnchor_1D_vertical(const UINT8 *imageIn, UINT8 *imageOut,
                                      int imageWidth, int imageHeight, int size)
    {
      const UINT8 *in, *inUp, *inDown, *current, *sentinel, *aux;
      UINT8 *out;
      UINT8 *outUp, *outDown;
      UINT8 min;
      int i, j, imageJump, sizeJump, sizeMinus1;
      int *histo, nbrBytes;
      int middle;

      in         = (UINT8 *) imageIn;
      out        = (UINT8 *) imageOut;
      imageJump  = imageWidth * (imageHeight - 1);
      sizeMinus1 = size - 1;
      sizeJump   = size * imageWidth;
      middle     = size / 2;

      /* Initialisation of the histogram */
      nbrBytes = 256 * sizeof(int);
      histo    = (int *) malloc(nbrBytes);

      /* Computation */
      /* Row by row */
      for (j = 0; j < imageWidth; j++) {
        /* Initialisation of both extremities of a column */
        inUp    = in + j;
        outUp   = out + j;
        inDown  = inUp + imageJump;
        outDown = outUp + imageJump;

        /* Handles the upper border */
        /* First half of the structuring element */
        memset(histo, 0, nbrBytes);
        min = *inUp;
        histo[min]++;
        for (i = 0; i < middle; i++) {
          inUp += imageWidth;
          histo[*inUp]++;
          if (*inUp < min) {
            min = *inUp;
          }
        }
        *outUp = min;

        /* Second half of the structuring element */
        for (i = 0; i < size - middle - 1; i++) {
          inUp += imageWidth;
          outUp += imageWidth;
          histo[*inUp]++;
          if (*inUp < min) {
            min = *inUp;
          }
          *outUp = min;
        }

        /* Uses the histogram as long as we have not found a new minimum */
        while ((inUp < inDown) && (min <= *(inUp + imageWidth))) {
          inUp += imageWidth;
          outUp += imageWidth;
          histo[*(inUp - sizeJump)]--;
          histo[*inUp]++;
          while (histo[min] <= 0) {
            min++;
          }
          *outUp = min;
        }

        /* Enters in the loop */
        min = *outUp;

      startLine:
        current = inUp + imageWidth;
        while ((current < inDown) && (*current <= min)) {
          min = *current;
          outUp += imageWidth;
          *outUp = min;
          current += imageWidth;
        }
        inUp     = current - imageWidth;
        sentinel = inUp + sizeJump;
        if (sentinel > inDown) {
          goto finishLine;
        }
        outUp += imageWidth;
        *outUp = min;

        /* We ran "size" pixels ahead */
        current += imageWidth;
        while (current < sentinel) {
          if (*current <= min) /* We have found a new minimum */
          {
            min = *current;
            outUp += imageWidth;
            *outUp = min;
            inUp   = current;
            goto startLine;
          }
          current += imageWidth;
          outUp += imageWidth;
          *outUp = min;
        }

        /* We did not find a smaller value in the segment in reach
         * of inUp; current is the first position outside the
         * reach of inUp
         */
        if (*current <= min) {
          min = *current;
          outUp += imageWidth;
          *outUp = min;
          inUp   = current;
          goto startLine;
        } else /* We can not avoid computing the histogram */
        {
          memset(histo, 0, nbrBytes);
          inUp += imageWidth;
          outUp += imageWidth;
          for (aux = inUp; aux <= current; aux += imageWidth) {
            histo[*aux]++;
          }
          min++;
          while (histo[min] <= 0) {
            min++;
          }
          *outUp = min;
        }

        /* We just follow the pixels, update the histogram and look for
         * the minimum */
        while (current < inDown) {
          current += imageWidth;
          if (*current <= min) {
            /* We have found a new mimum */
            min = *current;
            outUp += imageWidth;
            *outUp = min;
            inUp   = current;
            goto startLine;
          } else {
            /* Update the histogram */
            histo[*current]++;
            histo[*inUp]--;
            /* Recompute the minimum */
            while (histo[min] <= 0) {
              min++;
            }
            inUp += imageWidth;
            outUp += imageWidth;
            *outUp = min;
          }
        }

      finishLine:
        /* Handles the bottom border */
        /* First half of the structuring element */
        memset(histo, 0, nbrBytes);
        min = *inDown;
        histo[min]++;
        for (i = 0; i < middle; i++) {
          inDown -= imageWidth;
          histo[*inDown]++;
          if (*inDown < min) {
            min = *inDown;
          }
        }
        *outDown = min;

        /* Second half of the structuring element */
        for (i = 0; (i < size - middle - 1) && (outUp < outDown); i++) {
          inDown -= imageWidth;
          outDown -= imageWidth;
          histo[*inDown]++;
          if (*inDown < min) {
            min = *inDown;
          }
          *outDown = min;
        }

        /* Use the histogram as long as we have not found a new minimum */
        while (outUp < outDown) {
          inDown -= imageWidth;
          outDown -= imageWidth;
          histo[*(inDown + sizeJump)]--;
          histo[*inDown]++;
          if (*inDown < min) {
            min = *inDown;
          }
          while (histo[min] <= 0) {
            min++;
          }
          *outDown = min;
        }
      }

      /* Free memory */
      free(histo);
      return RES_OK;
    }

    /*!
    * \fn int dilationByAnchor_1D_horizontal(uint8_t *imageIn, uint8_t
    *imageOut, int imageWidth, int imageHeight, int size)
    * \param[in]  *imageIn Input buffer
    * \param[out]  *imageOut Output buffer
    * \param[in]  imageWidth Width of the image buffer
    * \param[in]  imageHeight Height of the image buffer
    * \param[in]  size (= width in tems of pixels) of the linear structuring
    element
    * \return Returns MORPHO_SUCCESS upon success, MORPHO_ERROR otherwise.
    *
    * \brief Dilation with an horizontal linear segment
    *
    * \ingroup libmorpho
    *
    * Dilation with an horizontal linear segment whose size is given in pixels.
    * For full technical details please refer to \ref detailsPage
    * or to
    * - M. Van Droogenbroeck and M. Buckley. <b>Morphological erosions and
    openings: fast algorithms based on anchors</b>. <em>Journal of Mathematical
    Imaging and Vision</em>, Special Issue on Mathematical Morphology after 40
    Years, 22(2-3):121-142, May 2005.
    *
    * \author      Marc Van Droogenbroeck
    */
    RES_C dilationByAnchor_1D_horizontal(const UINT8 *imageIn, UINT8 *imageOut,
                                         int imageWidth, int imageHeight,
                                         int size)
    {
      const UINT8 *in, *inLeft, *inRight, *current, *sentinel, *aux;
      UINT8 *out;
      UINT8 *outLeft, *outRight;
      UINT8 max;
      int i, j, imageWidthMinus1, sizeMinus1;
      int *histo, nbrBytes;
      int middle;

      in               = (UINT8 *) imageIn;
      out              = (UINT8 *) imageOut;
      imageWidthMinus1 = imageWidth - 1;
      sizeMinus1       = size - 1;
      middle           = size / 2;

      /* Initialisation of the histogram */
      nbrBytes = 256 * sizeof(int);
      histo    = (int *) malloc(nbrBytes);

      /* Computation */
      /* Row by row */
      for (j = 0; j < imageHeight; j++) {
        /* Initialisation of both extremities of a line */
        inLeft   = in + (j * imageWidth);
        outLeft  = out + (j * imageWidth);
        inRight  = inLeft + imageWidthMinus1;
        outRight = outLeft + imageWidthMinus1;

        /* Handles the left border */
        /* First half of the structuring element */
        memset(histo, 0, nbrBytes);
        max = *inLeft;
        histo[max]++;
        for (i = 0; i < middle; i++) {
          inLeft++;
          histo[*inLeft]++;
          if (*inLeft > max) {
            max = *inLeft;
          }
        }
        *outLeft = max;

        /* Second half of the structuring element */
        for (i = 0; i < size - middle - 1; i++) {
          inLeft++;
          outLeft++;
          histo[*inLeft]++;
          if (*inLeft > max) {
            max = *inLeft;
          }
          *outLeft = max;
        }

        /* Use the histogram as long as we have not found a new maximum */
        while ((inLeft < inRight) && (max >= *(inLeft + 1))) {
          inLeft++;
          outLeft++;
          histo[*(inLeft - size)]--;
          histo[*inLeft]++;
          while (histo[max] <= 0) {
            max--;
          }
          *outLeft = max;
        }

        /* Enters in the loop */
        max = *outLeft;

      startLine:
        current = inLeft + 1;
        while ((current < inRight) && (*current >= max)) {
          max = *current;
          outLeft++;
          *outLeft = max;
          current++;
        }
        inLeft   = current - 1;
        sentinel = inLeft + size;
        if (sentinel > inRight) {
          goto finishLine;
        }
        outLeft++;
        *outLeft = max;

        /* We ran "size" pixels ahead */
        current++;
        while (current < sentinel) {
          if (*current >= max) /* We have found a new maximum */
          {
            max = *current;
            outLeft++;
            *outLeft = max;
            inLeft   = current;
            goto startLine;
          }
          current++;
          outLeft++;
          *outLeft = max;
        }

        /* We did not find a smaller value in the segment in reach
         * of inLeft; current is the first position outside the
         * reach of inLeft
         */
        if (*current >= max) {
          max = *current;
          outLeft++;
          *outLeft = max;
          inLeft   = current;
          goto startLine;
        } else /* We can not avoid computing the histogram */
        {
          memset(histo, 0, nbrBytes);
          inLeft++;
          outLeft++;
          for (aux = inLeft; aux <= current; aux++) {
            histo[*aux]++;
          }
          max--;
          while (histo[max] <= 0) {
            max--;
          }
          *outLeft = max;
        }

        /* We just follow the pixels, update the histogram and look for
         * the maximum */
        while (current < inRight) {
          current++;
          if (*current >= max) {
            /* We have found a new mimum */
            max = *current;
            outLeft++;
            *outLeft = max;
            inLeft   = current;
            goto startLine;
          } else {
            /* Update the histogram */
            histo[*current]++;
            histo[*inLeft]--;
            /* Recompute the maximum */
            while (histo[max] <= 0) {
              max--;
            }
            inLeft++;
            outLeft++;
            *outLeft = max;
          }
        }

      finishLine:
        /* Handles the right border */
        /* First half of the structuring element */
        memset(histo, 0, nbrBytes);
        max = *inRight;
        histo[max]++;
        for (i = 0; i < middle; i++) {
          inRight--;
          histo[*inRight]++;
          if (*inRight > max) {
            max = *inRight;
          }
        }
        *outRight = max;

        /* Second half of the structuring element */
        for (i = 0; (i < size - middle - 1) && (outLeft < outRight); i++) {
          inRight--;
          outRight--;
          histo[*inRight]++;
          if (*inRight > max) {
            max = *inRight;
          }
          *outRight = max;
        }

        /* Use the histogram as long as we have not found a new maximum */
        while (outLeft < outRight) {
          inRight--;
          outRight--;
          histo[*(inRight + size)]--;
          histo[*inRight]++;
          if (*inRight > max) {
            max = *inRight;
          }
          while (histo[max] <= 0) {
            max--;
          }
          *outRight = max;
        }
      }

      /* Free memory */
      free(histo);

      return RES_OK;
    }

    /*!
    * \fn int dilationByAnchor_1D_vertical(uint8_t *imageIn, uint8_t *imageOut,
    int imageWidth, int imageHeight, int size)
    * \param[in]  *imageIn Input buffer
    * \param[out]  *imageOut Output buffer
    * \param[in]  imageWidth Width of the image buffer
    * \param[in]  imageHeight Height of the image buffer
    * \param[in]  size (= width in tems of pixels) of the linear structuring
    element
    * \return Returns MORPHO_SUCCESS upon success, MORPHO_ERROR otherwise.
    *
    * \brief Dilation with a vertical linear segment
    *
    * \ingroup libmorpho
    *
    * Dilation with a vertical linear segment whose size is given in pixels.
    * For full technical details please refer to \ref detailsPage
    * or to
    * - M. Van Droogenbroeck and M. Buckley. <b>Morphological erosions and
    openings: fast algorithms based on anchors</b>. <em>Journal of Mathematical
    Imaging and Vision</em>, Special Issue on Mathematical Morphology after 40
    Years, 22(2-3):121-142, May 2005.
    *
    * \author      Marc Van Droogenbroeck
    */
    RES_C dilationByAnchor_1D_vertical(const UINT8 *imageIn, UINT8 *imageOut,
                                       int imageWidth, int imageHeight,
                                       int size)
    {
      const UINT8 *in, *inUp, *inDown, *current, *sentinel, *aux;
      UINT8 *out;
      UINT8 *outUp, *outDown;
      UINT8 max;
      int i, j, imageJump, sizeJump, sizeMinus1;
      int *histo, nbrBytes;
      int middle;

      in         = (UINT8 *) imageIn;
      out        = (UINT8 *) imageOut;
      imageJump  = imageWidth * (imageHeight - 1);
      sizeMinus1 = size - 1;
      sizeJump   = size * imageWidth;
      middle     = size / 2;

      /* Initialisation of the histogram */
      nbrBytes = 256 * sizeof(int);
      histo    = (int *) malloc(nbrBytes);

      /* Computation */
      /* Row by row */
      for (j = 0; j < imageWidth; j++) {
        /* Initialisation of both extremities of a column */
        inUp    = in + j;
        outUp   = out + j;
        inDown  = inUp + imageJump;
        outDown = outUp + imageJump;

        /* Handles the upper border */
        /* First half of the structuring element */
        memset(histo, 0, nbrBytes);
        max = *inUp;
        histo[max]++;
        for (i = 0; i < middle; i++) {
          inUp += imageWidth;
          histo[*inUp]++;
          if (*inUp > max) {
            max = *inUp;
          }
        }
        *outUp = max;

        /* Second half of the structuring element */
        for (i = 0; i < size - middle - 1; i++) {
          inUp += imageWidth;
          outUp += imageWidth;
          histo[*inUp]++;
          if (*inUp > max) {
            max = *inUp;
          }
          *outUp = max;
        }

        /* Uses the histogram as long as we have not found a new maximum */
        while ((inUp < inDown) && (max >= *(inUp + imageWidth))) {
          inUp += imageWidth;
          outUp += imageWidth;
          histo[*(inUp - sizeJump)]--;
          histo[*inUp]++;
          while (histo[max] <= 0) {
            max--;
          }
          *outUp = max;
        }

        /* Enters in the loop */
        max = *outUp;

      startLine:
        current = inUp + imageWidth;
        while ((current < inDown) && (*current >= max)) {
          max = *current;
          outUp += imageWidth;
          *outUp = max;
          current += imageWidth;
        }
        inUp     = current - imageWidth;
        sentinel = inUp + sizeJump;
        if (sentinel > inDown) {
          goto finishLine;
        }
        outUp += imageWidth;
        *outUp = max;

        /* We ran "size" pixels ahead */
        current += imageWidth;
        while (current < sentinel) {
          if (*current >= max) /* We have found a new maximum */
          {
            max = *current;
            outUp += imageWidth;
            *outUp = max;
            inUp   = current;
            goto startLine;
          }
          current += imageWidth;
          outUp += imageWidth;
          *outUp = max;
        }

        /* We did not find a smaller value in the segment in reach
         * of inUp; current is the first position outside the
         * reach of inUp
         */
        if (*current >= max) {
          max = *current;
          outUp += imageWidth;
          *outUp = max;
          inUp   = current;
          goto startLine;
        } else /* We can not avoid computing the histogram */
        {
          memset(histo, 0, nbrBytes);
          inUp += imageWidth;
          outUp += imageWidth;
          for (aux = inUp; aux <= current; aux += imageWidth) {
            histo[*aux]++;
          }
          max--;
          while (histo[max] <= 0) {
            max--;
          }
          *outUp = max;
        }

        /* We just follow the pixels, update the histogram and look for
         * the maximum */
        while (current < inDown) {
          current += imageWidth;
          if (*current >= max) {
            /* We have found a new mimum */
            max = *current;
            outUp += imageWidth;
            *outUp = max;
            inUp   = current;
            goto startLine;
          } else {
            /* Update the histogram */
            histo[*current]++;
            histo[*inUp]--;
            /* Recompute the maximum */
            while (histo[max] <= 0) {
              max--;
            }
            inUp += imageWidth;
            outUp += imageWidth;
            *outUp = max;
          }
        }

      finishLine:
        /* Handles the bottom border */
        /* First half of the structuring element */
        memset(histo, 0, nbrBytes);
        max = *inDown;
        histo[max]++;
        for (i = 0; i < middle; i++) {
          inDown -= imageWidth;
          histo[*inDown]++;
          if (*inDown > max) {
            max = *inDown;
          }
        }
        *outDown = max;

        /* Second half of the structuring element */
        for (i = 0; (i < size - middle - 1) && (outUp < outDown); i++) {
          inDown -= imageWidth;
          outDown -= imageWidth;
          histo[*inDown]++;
          if (*inDown > max) {
            max = *inDown;
          }
          *outDown = max;
        }

        /* Use the histogram as long as we have not found a new maximum */
        while (outUp < outDown) {
          inDown -= imageWidth;
          outDown -= imageWidth;
          histo[*(inDown + sizeJump)]--;
          histo[*inDown]++;
          if (*inDown > max) {
            max = *inDown;
          }
          while (histo[max] <= 0) {
            max--;
          }
          *outDown = max;
        }
      }

      /* Free memory */
      free(histo);
      return RES_OK;
    }

    //
    // MORPH M INTERFACE
    //

    template <class T1, class T2>
    RES_C t_ImFastLineOpen_Anchors(const Image<T1> &imIn, const int radius,
                                   bool Horizontal, Image<T2> &imOut)
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

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();

      if (Horizontal)
        return openingByAnchor_1D_horizontal(bufferIn, bufferOut, W, H,
                                             2 * radius + 1);
      return openingByAnchor_1D_vertical(bufferIn, bufferOut, W, H,
                                         2 * radius + 1);
    }

    template <class T1, class T2>
    RES_C t_ImFastLineClose_Anchors(const Image<T1> &imIn, const int radius,
                                    bool Horizontal, Image<T2> &imOut)
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

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();

      if (Horizontal)
        return closingByAnchor_1D_horizontal(bufferIn, bufferOut, W, H,
                                             2 * radius + 1);
      return closingByAnchor_1D_vertical(bufferIn, bufferOut, W, H,
                                         2 * radius + 1);
    }

    template <class T1, class T2>
    RES_C t_ImFastLineDilate_Anchors(const Image<T1> &imIn, const int radius,
                                     bool Horizontal, Image<T2> &imOut)
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

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();

      if (Horizontal)
        return dilationByAnchor_1D_horizontal(bufferIn, bufferOut, W, H,
                                              2 * radius + 1);
      return dilationByAnchor_1D_vertical(bufferIn, bufferOut, W, H,
                                          2 * radius + 1);
    }

    template <class T1, class T2>
    RES_C t_ImFastLineErode_Anchors(const Image<T1> &imIn, const int radius,
                                    bool Horizontal, Image<T2> &imOut)
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

      int W, H;
      W                  = imIn.getWxSize();
      H                  = imIn.getWySize();
      const T1 *bufferIn = imIn.rawPointer();
      T2 *bufferOut      = imOut.rawPointer();

      if (Horizontal)
        return erosionByAnchor_1D_horizontal(bufferIn, bufferOut, W, H,
                                             2 * radius + 1);
      return erosionByAnchor_1D_vertical(bufferIn, bufferOut, W, H,
                                         2 * radius + 1);
    }

  } // namespace FastLine
} // namespace morphee

#endif
#endif
