/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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

#include "DColorMorpho.h"
#include "DColorConvert.h"

#include "NSTypes/RGB/include/DRGB.h"
#include "Morpho/include/private/DMorphImageOperations.hxx"

#include <complex>
#define _USE_MATH_DEFINES // for C++
// #include <math.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

namespace smil
{
  class labGrad_func : public MorphImageFunctionBase<RGB, UINT8>
  {
    ImDtTypes<UINT8>::lineType R, G, B;

    RES_T initialize(const imageInType &imIn, imageOutType &imOut,
                     const StrElt &se)
    {
      MorphImageFunctionBase<RGB, UINT8>::initialize(imIn, imOut, se);
      R = imIn.getPixels().arrays[0];
      G = imIn.getPixels().arrays[1];
      B = imIn.getPixels().arrays[2];

      return RES_OK;
    }

    double pwr2(double x)
    {
      return x * x;
    }

    void processPixel(size_t pointOffset, vector<int> &dOffsetList)
    {
      double distMax = 0;
      double dist    = 0;
      RGB    pVal    = pixelsIn[pointOffset];
      double r       = pVal.r;
      double g       = pVal.g;
      double b       = pVal.b;
      size_t dOff;

      vector<int>::iterator dOffset = dOffsetList.begin();
      while (dOffset != dOffsetList.end()) {
        dOff = pointOffset + *dOffset;
        dist = ((r - R[dOff]) * (r - R[dOff]) + (g - G[dOff]) * (g - G[dOff]) +
                (b - B[dOff]) * (b - B[dOff])) /
               3;
        // dist = (pwr2(r - R[dOff]) + pwr2(g - G[dOff]) + pwr2(b - B[dOff]) /
        // 3;
        //                  dist = pixelsIn[dOff].r;
        //          pixelsOut[pointOffset] = max(pixelsOut[pointOffset],
        //          pixelsIn[dOff]);
        if (dist > distMax)
          distMax = dist;
        dOffset++;
      }

      if (distMax > 0)
        dist = std::sqrt(distMax);

      pixelsOut[pointOffset] = dist;
    }
  };

  RES_T gradientLAB(const Image<RGB> &imIn, Image<UINT8> &imOut,
                    const StrElt &se, bool convertFirstToLAB)
  {
    ASSERT_ALLOCATED(&imIn);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    labGrad_func iFunc;

    if (convertFirstToLAB) {
      Image<RGB> tmpIm(imIn);
      RGBToLAB(imIn, tmpIm);
      return iFunc._exec(tmpIm, imOut, se);
    } else
      return iFunc._exec(imIn, imOut, se);
  }

  Image<UINT8> gradientLAB(const Image<RGB> &imIn, const StrElt &se,
                           bool convertFirstToLAB)
  {
    Image<UINT8> imOut(imIn);
    ASSERT(gradientLAB(imIn, imOut, se, convertFirstToLAB) == RES_OK, RES_ERR,
           imOut)
    return imOut;
  }

  class hlsGrad_func : public MorphImageFunctionBase<RGB, UINT8>
  {
    ImDtTypes<UINT8>::lineType H, L, S;

    RES_T initialize(const imageInType &imIn, imageOutType &imOut,
                     const StrElt &se)
    {
      MorphImageFunctionBase<RGB, UINT8>::initialize(imIn, imOut, se);
      H = imIn.getPixels().arrays[0];
      L = imIn.getPixels().arrays[1];
      S = imIn.getPixels().arrays[2];

      return RES_OK;
    }

    void processPixel(size_t pointOffset, vector<int> &dOffsetList)
    {
      double distMax = 0;
      RGB    pVal    = pixelsIn[pointOffset];
      double h       = double(pVal.r) * 2. * M_PI / 255.;
      double l       = double(pVal.g) / 255.;
      double s       = double(pVal.b) / 255.;
      // size_t dOff;

      // vector<int>::iterator dOffset = dOffsetList.begin();
      // while (dOffset != dOffsetList.end()) {
      for (auto dOffset = dOffsetList.begin(); dOffset != dOffsetList.end();
           dOffset++) {
        size_t dOff = pointOffset + *dOffset;
        double Hf   = double(H[dOff]) / 255. * 2 * M_PI; // Convert to radians
        double Lf   = double(L[dOff]) / 255.;
        double Sf   = double(S[dOff]) / 255.;

        // Calc. distance
        double d_delta_H;
        d_delta_H = std::fabs(h - Hf);

        // Circular H gradient. Check that we are between 0 and pi
        if (d_delta_H > M_PI) {
          d_delta_H = 2. * M_PI - d_delta_H;
        }
        // Normalize (-> [0, 1]) for luminace gradient consistency
        d_delta_H /= M_PI;

        double d_delta_L = std::fabs(l - Lf);
        double d_weight  = (s + Sf) / 2.;
        double dist;

        dist = d_weight * d_delta_H + (1 - d_weight) * d_delta_L;

        if (dist > distMax)
          distMax = dist;
        // dOffset++;
      }

      pixelsOut[pointOffset] = distMax * 255.;
    }
  };

  RES_T gradientHLS(const Image<RGB> &imIn, Image<UINT8> &imOut,
                    const StrElt &se, bool convertFirstToHLS)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    hlsGrad_func iFunc;

    if (convertFirstToHLS) {
      Image<RGB> tmpIm(imIn);
      RGBToHLS(imIn, tmpIm);
      return iFunc._exec(tmpIm, imOut, se);
    } else
      return iFunc._exec(imIn, imOut, se);
  }

  Image<UINT8> gradientHLS(const Image<RGB> &imIn, const StrElt &se,
                           bool convertFirstToHLS)
  {
    Image<UINT8> imOut(imIn);
    // ASSERT(gradientHLS(imIn, imOut, se, convertFirstToHLS) == RES_OK,
    // RES_ERR, imOut)
    RES_T r = gradientHLS(imIn, imOut, se, convertFirstToHLS);
    if (r != RES_OK) {
      ERR_MSG("Error evaluating gradientHLS");
    }

    return imOut;
  }

} // namespace smil
