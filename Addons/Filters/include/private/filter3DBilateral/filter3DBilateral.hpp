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

#ifndef _D_3D_BILATERAL_FILTER_HPP_
#define _D_3D_BILATERAL_FILTER_HPP_

#include "Core/include/private/DImage.hpp"
#include <stdlib.h>

namespace smil
{

  template <class T>
  RES_T recursiveBilateralFilter(const Image<T> &imIn,
                                 float sigmaW, float sigmaR, Image<T> &imOut)
  {
    int S[3];
    imOut.getSize(S);
    size_t nbrPixels         = S[0] * S[1] * S[2];
    size_t nbrPixelsPerSlice = S[0] * S[1];

    T *out = imOut.getPixels();
    T *in  = imIn.getPixels();

    UINT nthreads = Core::getInstance()->getNumberOfThreads();

    // Creating a range table.
    float alpha_f     = exp(-sqrt(2.0) / sigmaW);
    float inv_alpha_f = 1.f - alpha_f;

    /* JOE XXX
     * Sur cette partie du code :
     * * ca ne peut pas marcher avec des images de type non entier
     *   puisqu'on essaye d'allouer range_table avec une valeur
     *   non entiere
     * * pour des images a 32 ou 64 bits, on alloue des images 
     *   de taille "gigantesques" peut-etre sans en avoir besoin
     * * ne serait-il plus raisonnable que la dimension de range
     *   table soit plutot la valeur maximale de l'image ?
     *   A voir si ca ne pose pas d'autres problemes
     * * la boucle "for" :
     *   * la vérification de fin de boucle se fait avec un appel
     *     a fonction plutôt qu'a une valeur constante. Probleme
     *     de performance, surement
     */
    float inv_sigma_range = 1.f / sigmaR;
    float *range_table    = new float[ImDtTypes<T>::max() + 1];

    float ii = 0.f;
    for (size_t i = 0; i <= ImDtTypes<T>::max(); ++i) {
      range_table[i] = alpha_f * exp(ii * inv_sigma_range);
      ii -= 1.f;
    }

    // Create float buffers for exact computation.
    float *bufOut_causal    = new float[nbrPixels];
    float *bufFactor_causal = new float[nbrPixels];
    float *bufOut_anti      = new float[nbrPixels];
    float *bufFactor_anti   = new float[nbrPixels];
    float *bufOut_tmp       = new float[nbrPixels];

#pragma omp parallel num_threads(nthreads)
    {
      T *lineIn;
      T *linePreviousIn;
      T *lineOut;
      float *lineTmp;
      float *lineOut_causal;
      float *lineFac_causal;
      float *lineOut_anti;
      float *lineFac_anti;
      float *linePreviousOut;
      float *linePreviousFac;

      T *sliceIn;
      T *slicePreviousIn;
      T *sliceOut;
      float *sliceTmp;
      float *sliceOut_causal;
      float *sliceFac_causal;
      float *sliceOut_anti;
      float *sliceFac_anti;
      float *slicePreviousOut;
      float *slicePreviousFac;

#pragma omp for
      for (int z = 0; z < S[2]; ++z) {
        sliceIn         = &in[z * nbrPixelsPerSlice];
        sliceTmp        = &bufOut_tmp[z * nbrPixelsPerSlice];
        sliceOut_causal = &bufOut_causal[z * nbrPixelsPerSlice];
        sliceFac_causal = &bufFactor_causal[z * nbrPixelsPerSlice];
        sliceOut_anti   = &bufOut_anti[z * nbrPixelsPerSlice];
        sliceFac_anti   = &bufFactor_anti[z * nbrPixelsPerSlice];

        // Left and right
        for (int y = 0; y < S[1]; ++y) {
          lineIn         = &sliceIn[y * S[0]];
          lineOut_causal = &sliceOut_causal[y * S[0]];
          lineFac_causal = &sliceFac_causal[y * S[0]];

          // 1st pixel has no previous neighbour
          lineFac_causal[0] = 1.f;
          lineOut_causal[0] = lineIn[0];

          // From left to right
          for (int x = 1; x < S[0]; ++x) {
            float alpha       = range_table[labs(lineIn[x] - lineIn[x - 1])];
            lineFac_causal[x] = inv_alpha_f + alpha * lineFac_causal[x - 1];
            lineOut_causal[x] =
                inv_alpha_f * lineIn[x] + alpha * lineOut_causal[x - 1];
          }

          lineOut_anti = &sliceOut_anti[y * S[0]];
          lineFac_anti = &sliceFac_anti[y * S[0]];

          // Last pixel has no successive neighbour
          lineFac_anti[S[0] - 1] = 1.f;
          lineOut_anti[S[0] - 1] = lineIn[S[0] - 1];

          // From right to left
          for (int x = S[0] - 2; x >= 0; --x) {
            float alpha     = range_table[labs(lineIn[x] - lineIn[x + 1])];
            lineFac_anti[x] = inv_alpha_f + alpha * lineFac_anti[x + 1];
            lineOut_anti[x] =
                inv_alpha_f * lineIn[x] + alpha * lineOut_anti[x + 1];
          }

          lineTmp = &sliceTmp[y * S[0]];

          // Store the result in tmp
          for (int x = 0; x < S[0]; ++x) {
            float factor = 1.f / (lineFac_causal[x] + lineFac_anti[x]);
            lineTmp[x]   = factor * (lineOut_causal[x] + lineOut_anti[x]);
          }
        }

        // From top to bottom
        // First line done separately
        lineIn         = &sliceIn[0];
        lineOut_causal = &sliceOut_causal[0];
        lineFac_causal = &sliceFac_causal[0];
        lineTmp        = &sliceTmp[0];

        for (int x = 0; x < S[0]; ++x) {
          lineFac_causal[x] = 1.f;
          lineOut_causal[x] = lineTmp[x];
        }

        for (int y = 1; y < S[1]; ++y) {
          lineIn          = &sliceIn[y * S[0]];
          lineOut_causal  = &sliceOut_causal[y * S[0]];
          lineFac_causal  = &sliceFac_causal[y * S[0]];
          lineTmp         = &sliceTmp[y * S[0]];
          linePreviousIn  = &sliceIn[(y - 1) * S[0]];
          linePreviousOut = &sliceOut_causal[(y - 1) * S[0]];
          linePreviousFac = &sliceFac_causal[(y - 1) * S[0]];

          for (int x = 0; x < S[0]; ++x) {
            float alpha = range_table[labs(lineIn[x] - linePreviousIn[x])];
            lineFac_causal[x] = inv_alpha_f + alpha * linePreviousFac[x];
            lineOut_causal[x] =
                inv_alpha_f * lineTmp[x] + alpha * linePreviousOut[x];
          }
        }

        lineIn       = &sliceIn[S[1] - 1];
        lineOut_anti = &sliceOut_anti[S[1] - 1];
        lineFac_anti = &sliceFac_anti[S[1] - 1];
        lineTmp      = &sliceTmp[S[1] - 1];

        // From bottom to top
        // First line done separately
        for (int x = 0; x < S[0]; ++x) {
          lineFac_anti[x] = 1.f;
          lineOut_anti[x] = lineTmp[x];
        }

        for (int y = S[1] - 2; y >= 0; --y) {
          lineIn          = &sliceIn[y * S[0]];
          lineOut_anti    = &sliceOut_anti[y * S[0]];
          lineFac_anti    = &sliceFac_anti[y * S[0]];
          lineTmp         = &sliceTmp[y * S[0]];
          linePreviousIn  = &sliceIn[(y + 1) * S[0]];
          linePreviousOut = &sliceOut_anti[(y + 1) * S[0]];
          linePreviousFac = &sliceFac_anti[(y + 1) * S[0]];

          for (int x = 0; x < S[0]; ++x) {
            float alpha     = range_table[labs(lineIn[x] - linePreviousIn[x])];
            lineFac_anti[x] = inv_alpha_f + alpha * linePreviousFac[x];
            lineOut_anti[x] =
                inv_alpha_f * lineTmp[x] + alpha * linePreviousOut[x];
          }
        }

        // Store the result in tmp
        for (int y = 0; y < S[1]; ++y) {
          lineOut_causal = &sliceOut_causal[y * S[0]];
          lineOut_anti   = &sliceOut_anti[y * S[0]];
          lineFac_causal = &sliceFac_causal[y * S[0]];
          lineFac_anti   = &sliceFac_anti[y * S[0]];
          lineTmp        = &sliceTmp[y * S[0]];

          for (int x = 0; x < S[0]; ++x) {
            float factor = 1.f / (lineFac_causal[x] + lineFac_anti[x]);
            lineTmp[x]   = T(factor * (lineOut_causal[x] + lineOut_anti[x]));
          }
        }
      }

      // From front to back
      // First slide done separately
      sliceTmp        = &bufOut_tmp[0];
      sliceOut_causal = &bufOut_causal[0];
      sliceFac_causal = &bufFactor_causal[0];

#pragma omp for
      for (int y = 0; y < S[1]; ++y) {
        lineTmp        = &sliceTmp[y * S[0]];
        lineOut_causal = &sliceOut_causal[y * S[0]];
        lineFac_causal = &sliceFac_causal[y * S[0]];

        for (int x = 0; x < S[0]; ++x) {
          lineFac_causal[x] = 1.f;
          lineOut_causal[x] = lineTmp[x];
        }
      }

      for (int z = 1; z < S[2]; ++z) {
        sliceIn          = &in[z * nbrPixelsPerSlice];
        sliceTmp         = &bufOut_tmp[z * nbrPixelsPerSlice];
        sliceOut_causal  = &bufOut_causal[z * nbrPixelsPerSlice];
        sliceFac_causal  = &bufFactor_causal[z * nbrPixelsPerSlice];
        slicePreviousIn  = &in[(z - 1) * nbrPixelsPerSlice];
        slicePreviousOut = &bufOut_causal[(z - 1) * nbrPixelsPerSlice];
        slicePreviousFac = &bufFactor_causal[(z - 1) * nbrPixelsPerSlice];

#pragma omp for
        for (int y = 0; y < S[1]; ++y) {
          lineIn          = &sliceIn[y * S[0]];
          lineTmp         = &sliceTmp[y * S[0]];
          lineOut_causal  = &sliceOut_causal[y * S[0]];
          lineFac_causal  = &sliceFac_causal[y * S[0]];
          linePreviousIn  = &slicePreviousIn[y * S[0]];
          linePreviousOut = &slicePreviousOut[y * S[0]];
          linePreviousFac = &slicePreviousFac[y * S[0]];

          for (int x = 0; x < S[0]; ++x) {
            float alpha = range_table[labs(lineIn[x] - linePreviousIn[x])];
            lineFac_causal[x] = inv_alpha_f + alpha * linePreviousFac[x];
            lineOut_causal[x] =
                inv_alpha_f * lineTmp[x] + alpha * linePreviousOut[x];
          }
        }
      }

      // From back to front
      // Last slide done separately
      sliceTmp      = &bufOut_tmp[S[2] - 1];
      sliceOut_anti = &bufOut_anti[S[2] - 1];
      sliceFac_anti = &bufFactor_anti[S[2] - 1];

#pragma omp for
      for (int y = 0; y < S[1]; ++y) {
        lineTmp      = &sliceTmp[y * S[0]];
        lineOut_anti = &sliceOut_anti[y * S[0]];
        lineFac_anti = &sliceFac_anti[y * S[0]];

        for (int x = 0; x < S[0]; ++x) {
          lineFac_anti[x] = 1.f;
          lineOut_anti[x] = lineTmp[x];
        }
      }

      for (int z = S[2] - 2; z >= 0; --z) {
        sliceIn          = &in[z * nbrPixelsPerSlice];
        sliceTmp         = &bufOut_tmp[z * nbrPixelsPerSlice];
        sliceOut_anti    = &bufOut_anti[z * nbrPixelsPerSlice];
        sliceFac_anti    = &bufFactor_anti[z * nbrPixelsPerSlice];
        slicePreviousIn  = &in[(z + 1) * nbrPixelsPerSlice];
        slicePreviousOut = &bufOut_anti[(z + 1) * nbrPixelsPerSlice];
        slicePreviousFac = &bufFactor_anti[(z + 1) * nbrPixelsPerSlice];

#pragma omp for
        for (int y = 0; y < S[1]; ++y) {
          lineIn          = &sliceIn[y * S[0]];
          lineTmp         = &sliceTmp[y * S[0]];
          lineOut_anti    = &sliceOut_anti[y * S[0]];
          lineFac_anti    = &sliceFac_anti[y * S[0]];
          linePreviousIn  = &slicePreviousIn[y * S[0]];
          linePreviousOut = &slicePreviousOut[y * S[0]];
          linePreviousFac = &slicePreviousFac[y * S[0]];

          for (int x = 0; x < S[0]; ++x) {
            float alpha     = range_table[labs(lineIn[x] - linePreviousIn[x])];
            lineFac_anti[x] = inv_alpha_f + alpha * linePreviousFac[x];
            lineOut_anti[x] =
                inv_alpha_f * lineTmp[x] + alpha * linePreviousOut[x];
          }
        }
      }

      for (int z = 0; z < S[2]; ++z) {
        sliceOut_causal = &bufOut_causal[z * nbrPixelsPerSlice];
        sliceFac_causal = &bufFactor_causal[z * nbrPixelsPerSlice];
        sliceOut_anti   = &bufOut_anti[z * nbrPixelsPerSlice];
        sliceFac_anti   = &bufFactor_anti[z * nbrPixelsPerSlice];
        sliceOut        = &out[z * nbrPixelsPerSlice];

#pragma omp for
        for (int y = 0; y < S[1]; ++y) {
          lineOut_causal = &sliceOut_causal[y * S[0]];
          lineFac_causal = &sliceFac_causal[y * S[0]];
          lineOut_anti   = &sliceOut_anti[y * S[0]];
          lineFac_anti   = &sliceFac_anti[y * S[0]];
          lineOut        = &sliceOut[y * S[0]];

          // Storing the final result
          for (int x = 0; x < S[0]; ++x) {
            float factor = 1.f / (lineFac_causal[x] + lineFac_anti[x]);
            lineOut[x]   = T(factor * (lineOut_causal[x] + lineOut_anti[x]));
          }
        }
      }
    }
    delete[] range_table;
    delete[] bufOut_causal;
    delete[] bufOut_anti;
    delete[] bufFactor_causal;
    delete[] bufFactor_anti;
    delete[] bufOut_tmp;

    return RES_OK;
  }

} // namespace smil

#endif // _D_3D_BILATERAL_FILTER_HPP_
