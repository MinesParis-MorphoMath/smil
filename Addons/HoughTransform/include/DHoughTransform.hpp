/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _D_HOUGH_TRANSFORM_HPP
#define _D_HOUGH_TRANSFORM_HPP

#include "Core/include/private/DImage.hxx"
#include "Base/include/DBase.h"

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup AddonHoughTransform Features detection (Hough)
   * @{
   */

  /**
   * Hough Lines
   * @brief houghLines
   *
   * @param[in]  imIn - input image
   * @param[in]  thetaRes - angle resolution - scale in fraction of 180 degres
   * @param[in]  rhoRes - length resolution - fraction of maximum distance in
   *   the image
   * @param[out] imOut - output image
   *
   */
  template <class T1, class T2>
  RES_T houghLines(Image<T1> &imIn, double thetaRes, double rhoRes,
                   Image<T2> &imOut)
  {
    size_t wIn = imIn.getWidth();
    size_t hIn = imIn.getHeight();

    double rhoMax = std::sqrt(wIn * wIn + hIn * hIn);

    size_t wOut = thetaRes * 180;
    size_t hOut = rhoRes * rhoMax;

    ImageFreezer freeze(imOut);
    imOut.setSize(wOut, hOut);
    fill(imOut, T2(0));

    typename Image<T1>::sliceType linesIn  = imIn.getLines();
    typename Image<T2>::sliceType linesOut = imOut.getLines();
    typename Image<T1>::lineType  lIn;

    double thetaStep = PI / wOut;
    double rhoStep   = rhoMax / hOut;

    for (off_t j = 0; j < (off_t) hIn; j++) {
      lIn = linesIn[j];

      double coef = 1. / rhoStep;
      for (off_t i = 0; i < (off_t) wIn; i++) {
        if (lIn[i] != 0) {
#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
          for (off_t t = 0; t < (off_t) wOut; t++) {
            double theta = t * thetaStep;
            double rho   = coef * (i * cos(theta) + j * sin(theta));

            if (rho >= 0 && rho < hOut) {
              off_t irho = int(fabs(rho));
              linesOut[irho][t] += 1;
            }
          }
        }
      }
    }
    return RES_OK;
  }

  /**
   * Hough Circles
   * @brief Hough Circles (2D Images)
   * @param[in]  imIn : input image
   * @param[in]  xyResol : distance resolution for circle center
   * @param[in]  rhoResol : distance resolution for radius
   * @param[out] imOut : output image with Hough Transform for circles
   *
   */
  template <class T1, class T2>
  RES_T houghCircles(Image<T1> &imIn, double xyResol, double rhoResol,
                     Image<T2> &imOut)
  {
    size_t wIn = imIn.getWidth();
    size_t hIn = imIn.getHeight();

    double rhoMax = std::sqrt(wIn * wIn + hIn * hIn);

    size_t wOut = xyResol * wIn;
    size_t hOut = xyResol * hIn;
    off_t  dOut = rhoResol * rhoMax;

    ImageFreezer freeze(imOut);
    imOut.setSize(wOut, hOut, dOut);
    fill(imOut, T2(0));

    typename Image<T1>::sliceType linesIn = imIn.getSlices()[0];
    typename Image<T1>::lineType  lIn;
    typename Image<T2>::volType   slicesOut = imOut.getSlices();

    off_t  rho;
    double coef = 1 / (xyResol * rhoResol);

    for (off_t j = 0; j < (off_t) hIn; j++) {
      lIn = linesIn[j];
      for (off_t i = 0; i < (off_t) wIn; i++) {
        if (lIn[i] != T1(0)) {
#ifdef USE_OPEN_MP
#pragma omp for
#endif // USE_OPEN_MP
          for (off_t j2 = 0; j2 < (off_t) hOut; j2++) {
            for (off_t i2 = 0; i2 < (off_t) wOut; i2++) {
              if (i != i2 && j != j2) {
                rho = coef * std::sqrt((i * xyResol - i2) * (i * xyResol - i2) +
                                       (j * xyResol - j2) * (j * xyResol - j2));
                if (rho >= 0 && rho < dOut)
                  slicesOut[rho][j2][i2] += 1;
              }
            }
          }
        }
      }
    }
    return RES_OK;
  }

  /**
   * Hough Circles
   *
   */
  template <class T1, class T2>
  RES_T houghCircles(Image<T1> &imIn, double resol, Image<T2> &imOut,
                     Image<T2> &imRadiiOut)
  {
    size_t wIn = imIn.getWidth();
    size_t hIn = imIn.getHeight();

    // Commente - rhoMax non utilise
    // double rhoMax = sqrt(wIn*wIn + hIn*hIn);

    size_t wOut = resol * wIn;
    size_t hOut = resol * hIn;

    ImageFreezer freeze(imRadiiOut);

    Image<double> imCentersOut(wOut, hOut);
    imRadiiOut.setSize(wOut, hOut);

    fill(imCentersOut, 0.);
    fill(imRadiiOut, T2(0));

    typename Image<T1>::sliceType     linesIn = imIn.getSlices()[0];
    typename Image<T1>::lineType      lIn;
    typename Image<double>::sliceType linesOut1 = imCentersOut.getSlices()[0];
    typename Image<T2>::sliceType     linesOut2 = imRadiiOut.getSlices()[0];

    double rho;
    size_t nonZeroPts = 0;

    for (off_t j = 0; j < (off_t) imIn.getHeight(); j++) {
      lIn = linesIn[j];
      for (off_t i = 0; i < (off_t) imIn.getWidth(); i++) {
        if (lIn[i] != T1(0)) {
          nonZeroPts++;
          for (off_t j2 = 0; j2 < (off_t) hOut; j2++) {
            for (off_t i2 = 0; i2 < (off_t) wOut; i2++) {
              if (i != i2 && j != j2) {
                rho = std::sqrt(double((i * resol - i2) * (i * resol - i2) +
                                       (j * resol - j2) * (j * resol - j2)));
                // JOE : Why 100 and 500 ???
                if (rho > 100 && rho < 500) {
                  linesOut1[j2][i2] += 1;
                  if (linesOut1[j2][i2] > nonZeroPts &&
                      linesOut2[j2][i2] < T2(rho))
                    linesOut2[j2][i2] = T2(rho);
                }
              }
            }
          }
        }
      }
    }
    stretchHistogram(imCentersOut, imOut);
    return RES_OK;
  }

  /**@}*/
} // namespace smil

#endif // _D_HOUGH_TRANSFORM_HPP
