/*
 * __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2022, Centre de Morphologie Mathematique
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
 *   Parsimonious incomplete-path openings
 *   Author : Petr Dokladal (based on V. Morard's code)
 *   implements Parsimonious Incomplete-Path Openings using the Rank Opening
 *   (see Pierre Soille, On morphological operators based on rank filters,
 *    Pattern Recognition 35 (2002) 527-535)
 *   the rank opening is implemented as a rank filter followed by a dilation.
 *   It uses histograms.
 *   TO DOS : 1) make it isotropic.
 *            2) extend the border mirroring to SE>n. Will be useful for short
 *            diagonal paths in the corners of the image
 *               (see how this extension can be done)
 *   This file does... some very complex morphological operation...
 *
 * History :
 *   - 08/06/2020 - by Jose-Marcio Martins da Cruz
 *     Porting from Petr Dokladal code in Morph-M
 *
 * __HEAD__ - Stop here !
 */

#ifndef __FAST_PIPATH_OPENING_HPP__
#define __FAST_PIPATH_OPENING_HPP__

#include <queue>
#include "Morpho/include/DMorpho.h"

namespace smil
{
  /*
   *
   *
   *
   */
  template <class T> class ParsimoniousPathOpeningClass
  {
    // better to define this as a boolean
    bool MIRROR_BORDERS = true;

  public:
    ParsimoniousPathOpeningClass()
    {
      MIRROR_BORDERS = true;
    }
    ~ParsimoniousPathOpeningClass()
    {
    }

  private:

    /*
     *
     */
    T rank(T *H, int r)
    {
      int cnt = 0;
      T ii;
      /*
       * size of HWidth shall be bigger than size of T
       * this loop may take really long when T is of type UINT32
       */
      off_t HWidth = numeric_limits<T>::max();

      for (ii = 0; ii < HWidth; ii++) {
        cnt += H[ii];
        if (cnt > r)
          break;
      }
      return ii;
    }

    /*
     *
     */
    RES_T rank_filter_indx(T *x, off_t *indx, int n, int SE, int r, T *y)
    {
      // cout << "rank_filter_indx " << n << " " << SE << " " << r << endl;
      uint64_t HWidth = numeric_limits<T>::max();

      T *H = new T[HWidth]();

      //  pad border
      off_t *indx_pad = new off_t[n + 2 * SE]();
      memcpy(indx_pad + SE, indx, n * sizeof(off_t));

      if (!MIRROR_BORDERS) {
        //  pad border by replicating
        for (off_t ii = 0; ii < SE; ii++) {
          indx_pad[ii]          = indx[0];
          indx_pad[n + SE + ii] = indx[n - 1];
        }
      } else {
        //  pad border by mirroring
        for (off_t ii = 0; ii < SE; ii++) {
          off_t ix, iy;
          ix = ii;
          iy = (SE - ii) % n;
          indx_pad[ix] = indx[iy];
          ix = SE + n + ii;
          iy = n - 2 - ii;
          while (iy < 0)
            iy += n;
          iy = iy % n;
          indx_pad[ix] = indx[iy];
        }
      }

      // init histogram
      for (off_t ii = 0; ii < SE; ii++) {
        // no weight is used
        H[x[indx_pad[ii + SE - SE / 2]]] += 1;
      }
      y[0] = rank(H, r);

      for (off_t ii = 0; ii < n - 1; ii++) {
        // no weight is used
        H[x[indx_pad[ii + SE - SE / 2]]] -= 1;
        H[x[indx_pad[ii + 2 * SE - SE / 2]]] += 1;
        y[ii + 1] = rank(H, r);
      }

      delete[] indx_pad;
      delete[] H;

      return RES_OK;
    }

    /*
     *
     */
    RES_T conj_dilation(T *x, int n, int SE, T *y)
    {
      uint64_t HWidth = numeric_limits<T>::max();
      T *H            = new T[HWidth]();
      //  pad border by replicating
      T *x_pad = new T[n + 2 * SE]();
      memcpy(x_pad + SE, x, n * sizeof(T));
      for (off_t i = 0; i < SE; i++) {
        x_pad[i]          = x[0];
        x_pad[n + SE + i] = x[n - 1];
      }

      // init histogram
      for (off_t ii = 0; ii < SE; ii++)
        H[x_pad[ii + SE - SE / 2 + (SE + 1) % 2]] += 1;
      y[0] = rank(H, SE - 1);

      for (off_t ii = 0; ii < n - 1; ii++) {
        H[x_pad[ii + SE - SE / 2 + (SE + 1) % 2]] -= 1;
        H[x_pad[ii + 2 * SE - SE / 2 + (SE + 1) % 2]] += 1;
        y[ii + 1] = rank(H, SE - 1);
      }

      delete[] x_pad;
      delete[] H;

      return RES_OK;
    }

    /*
     *
     */
    RES_T rank_open(T *x, off_t *indx, int n, int SE, int r, T *y)
    {
      T *xi  = new T[n]();
      T *dxi = new T[n]();

      (void ) rank_filter_indx(x, indx, n, SE, r, xi);
      (void ) conj_dilation(xi, n, SE, dxi);

      for (off_t ii = 0; ii < n; ii++)
        y[indx[ii]] = max(y[indx[ii]], min(x[indx[ii]], dxi[ii]));

      delete[] xi;
      delete[] dxi;

      return RES_OK;
    }

  public:
    /*
     *
     *
     */
    RES_T doIt(Image<T> &imIn, int size, int tolerance, int step,
               Image<T> &imOut)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);
      ASSERT(imIn.getDepth() == 1);
      ASSERT(tolerance >= 0);
      ASSERT(step > 0);

      int Dir, DirH, DirV;
      int i, j, whichLine;

      off_t W = (off_t) imIn.getWidth();
      off_t H = (off_t) imIn.getHeight();
      T F     = 0;

      typename ImDtTypes<T>::lineType bufferIn  = imIn.getPixels();
      typename ImDtTypes<T>::lineType bufferOut = imOut.getPixels();

      // Initialisation
      fill(imOut, T(0));
      off_t *indx = new off_t[W + H]();

      // First direction Left to right
      // printf ("direction 1: left-to-right\n");
      for (j = 0; j < H; j += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = j * W;
        whichLine = j;

        for (i = 1; i < W; i++) {
          F       = bufferIn[i + whichLine * W];
          indx[i] = i + whichLine * W;
          Dir     = 0;
          // JOE lengthDir = 1;
          if (whichLine - 1 >= 0 && F < bufferIn[i + (whichLine - 1) * W]) {
            F       = bufferIn[i + (whichLine - 1) * W];
            indx[i] = i + (whichLine - 1) * W;
            Dir     = -1;
            // JOE lengthDir = SQRT_2;
          }
          if (whichLine + 1 < H && F < bufferIn[i + (whichLine + 1) * W]) {
            indx[i] = i + (whichLine + 1) * W;
            Dir     = 1;
            // JOE lengthDir = SQRT_2;
          }
          whichLine += Dir;
        }
        rank_open(bufferIn, indx, W, size, tolerance, bufferOut);
      }

      // Second direction Right to left
      // printf ("direction 2: right-to-left\n");
      for (j = 0; j < H; j += step) {
        // JOE wp          = 0;
        // JOE Length      = 0;
        // JOE stackSize   = 0;
        indx[W - 1] = W - 1 + j * W;
        whichLine   = j;
        for (i = W - 2; i >= 0; i--) {
          F       = bufferIn[i + whichLine * W];
          indx[i] = i + whichLine * W;
          Dir     = 0;
          // JOE lengthDir = 1;
          if (whichLine - 1 >= 0 && F < bufferIn[i + (whichLine - 1) * W]) {
            F       = bufferIn[i + (whichLine - 1) * W];
            indx[i] = i + (whichLine - 1) * W;
            Dir     = -1;
            // JOE lengthDir = SQRT_2;
          }
          if (whichLine + 1 < H && F < bufferIn[i + (whichLine + 1) * W]) {
            indx[i] = i + (whichLine + 1) * W;
            Dir     = 1;
            // JOE lengthDir = SQRT_2;
          }
          whichLine += Dir;
        }
        rank_open(bufferIn, indx, W, size, tolerance, bufferOut);
      }

      // Third direction Top to bottom
      // printf ("direction 3: Top to bottom\n");
      for (i = 0; i < W; i += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = i;
        whichLine = i;
        for (j = 1; j < H; j++) {
          F       = bufferIn[whichLine + j * W];
          indx[j] = whichLine + j * W;
          Dir     = 0;
          // JOE lengthDir = 1;
          if (whichLine - 1 >= 0 && F < bufferIn[whichLine - 1 + j * W]) {
            F       = bufferIn[whichLine - 1 + j * W];
            indx[j] = whichLine - 1 + j * W;
            Dir     = -1;
            // JOE lengthDir = SQRT_2;
          }
          if (whichLine + 1 < W && F < bufferIn[whichLine + 1 + j * W]) {
            indx[j] = whichLine + 1 + j * W;
            Dir     = 1;
            // JOE lengthDir = SQRT_2;
          }
          whichLine += Dir;
        }
        rank_open(bufferIn, indx, H, size, tolerance, bufferOut);
      }

      // Fourth direction bottom to top
      // printf ("direction 4: bottom to top\n");
      for (i = 0; i < W; i += step) {
        // JOE wp          = 0;
        // JOE Length      = 0;
        // JOE stackSize   = 0;
        indx[H - 1] = i + (H - 1) * W;
        whichLine   = i;
        for (j = H - 2; j >= 0; j--) {
          F       = bufferIn[whichLine + j * W];
          indx[j] = whichLine + j * W;
          Dir     = 0;
          // JOE lengthDir = 1;
          if (whichLine - 1 >= 0 && F < bufferIn[whichLine - 1 + j * W]) {
            F       = bufferIn[whichLine - 1 + j * W];
            indx[j] = whichLine - 1 + j * W;
            Dir     = -1;
            // JOE lengthDir = SQRT_2;
          }
          if (whichLine + 1 < W && F < bufferIn[whichLine + 1 + j * W]) {
            indx[j] = whichLine + 1 + j * W;
            Dir     = 1;
            // JOE lengthDir = SQRT_2;
          }
          whichLine += Dir;
        }
        rank_open(bufferIn, indx, H, size, tolerance, bufferOut);
      }

      // Fifth direction bottom left, top right
      // printf ("direction 5: bottom left, top right\n");
      for (j = 1; j < H; j += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = j * W;
        whichLine = j;
        i         = 0;
        int cnt   = 1;

        do {
          F         = bufferIn[i + 1 + (whichLine - 1) * W];
          indx[cnt] = i + 1 + (whichLine - 1) * W;
          DirV      = -1;
          DirH      = 1;
          // JOE lengthDir = SQRT_2;
          if (whichLine - 1 >= 0 && F < bufferIn[i + (whichLine - 1) * W]) {
            F         = bufferIn[i + (whichLine - 1) * W];
            indx[cnt] = i + (whichLine - 1) * W;
            DirH      = 0;
            DirV      = -1;
            // JOE lengthDir = 1;
          }
          if (i + 1 < W && F < bufferIn[i + 1 + (whichLine) *W]) {
            F         = bufferIn[i + 1 + (whichLine) *W];
            indx[cnt] = i + 1 + (whichLine) *W;
            DirH      = 1;
            DirV      = 0;
            // JOE lengthDir = 1;
          }
          i += DirH;
          whichLine += DirV;
          cnt++;
        } while (i <= W - 2 && whichLine >= 1);
        // cout << "Cinq" << endl;
        rank_open(bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

      for (i = 0; i < W - 1; i += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = i + (H - 1) * W;
        whichLine = i;
        j         = H - 1;
        int cnt   = 1;

        do {
          F         = bufferIn[whichLine + 1 + (j - 1) * W];
          indx[cnt] = whichLine + 1 + (j - 1) * W;
          DirV      = -1;
          DirH      = 1;
          // JOE lengthDir = SQRT_2;
          if (j - 1 >= 0 && F < bufferIn[whichLine + (j - 1) * W]) {
            F         = bufferIn[whichLine + (j - 1) * W];
            indx[cnt] = whichLine + (j - 1) * W;
            DirH      = 0;
            DirV      = -1;
            // JOE lengthDir = 1;
          }
          if (whichLine + 1 < W && F < bufferIn[whichLine + 1 + (j) *W]) {
            F         = bufferIn[whichLine + 1 + (j) *W];
            indx[cnt] = whichLine + 1 + (j) *W;
            DirH      = 1;
            DirV      = 0;
            // JOE lengthDir = 1;
          }
          whichLine += DirH;
          j += DirV;
          cnt++;
        } while (whichLine <= W - 2 && j >= 1);
        rank_open(bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

      // Sixth direction top right to bottom left
      // printf ("direction 6: top right to bottom left\n");
      for (j = 0; j < H - 1; j += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = W - 1 + j * W;
        whichLine = j;
        i         = W - 1;
        int cnt   = 1;
        do {
          F         = bufferIn[i - 1 + (whichLine + 1) * W];
          indx[cnt] = i - 1 + (whichLine + 1) * W;
          DirV      = 1;
          DirH      = -1;
          // JOE lengthDir = SQRT_2;
          if (whichLine + 1 < H && F < bufferIn[i + (whichLine + 1) * W]) {
            F         = bufferIn[i + (whichLine + 1) * W];
            indx[cnt] = i + (whichLine + 1) * W;
            DirH      = 0;
            DirV      = 1;
            // JOE lengthDir = 1;
          }
          if (i - 1 >= 0 && F < bufferIn[i - 1 + (whichLine) *W]) {
            F         = bufferIn[i - 1 + (whichLine) *W];
            indx[cnt] = i - 1 + (whichLine) *W;
            DirH      = -1;
            DirV      = 0;
            // JOE lengthDir = 1;
          }
          i += DirH;
          whichLine += DirV;
          cnt++;
        } while (i >= 1 && whichLine <= H - 2);
        rank_open(bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

      for (i = 1; i < W; i += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = i;
        whichLine = i;
        j         = 0;
        int cnt   = 1;

        do {
          F         = bufferIn[whichLine - 1 + (j + 1) * W];
          indx[cnt] = whichLine - 1 + (j + 1) * W;
          DirV      = 1;
          DirH      = -1;
          // JOE lengthDir = SQRT_2;
          if (j + 1 < H && F < bufferIn[whichLine + (j + 1) * W]) {
            F         = bufferIn[whichLine + (j + 1) * W];
            indx[cnt] = whichLine + (j + 1) * W;
            DirH      = 0;
            DirV      = 1;
            // JOE lengthDir = 1;
          }
          if (whichLine - 1 >= 0 && F < bufferIn[whichLine - 1 + (j) *W]) {
            F         = bufferIn[whichLine - 1 + (j) *W];
            indx[cnt] = whichLine - 1 + (j) *W;
            DirH      = -1;
            DirV      = 0;
            // JOE lengthDir = 1;
          }
          whichLine += DirH;
          j += DirV;
          cnt++;
        } while (whichLine >= 1 && j <= H - 2);
        rank_open(bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

      // Seventh direction top left, bottom right
      // printf ("direction 7: top left, bottom right\n");
      for (j = 0; j < H - 1; j += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = j * W;
        whichLine = j;
        i         = 0;
        int cnt   = 1;

        do {
          F         = bufferIn[i + 1 + (whichLine + 1) * W];
          indx[cnt] = i + 1 + (whichLine + 1) * W;
          DirV      = 1;
          DirH      = 1;
          // JOE lengthDir = SQRT_2;
          if (whichLine + 1 < H && F < bufferIn[i + (whichLine + 1) * W]) {
            F         = bufferIn[i + (whichLine + 1) * W];
            indx[cnt] = i + (whichLine + 1) * W;
            DirH      = 0;
            DirV      = 1;
            // JOE lengthDir = 1;
          }
          if (i + 1 < W && F < bufferIn[i + 1 + (whichLine) *W]) {
            F         = bufferIn[i + 1 + (whichLine) *W];
            indx[cnt] = i + 1 + (whichLine) *W;
            DirH      = 1;
            DirV      = 0;
            // JOE lengthDir = 1;
          }
          i += DirH;
          whichLine += DirV;
          cnt++;
        } while (i <= W - 2 && whichLine <= H - 2);
        rank_open(bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

      for (i = 0; i < W - 1; i += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = i;
        whichLine = i;
        j         = 0;
        int cnt   = 1;

        do {
          F         = bufferIn[whichLine + 1 + (j + 1) * W];
          indx[cnt] = whichLine + 1 + (j + 1) * W;
          DirV      = 1;
          DirH      = 1;
          // JOE lengthDir = SQRT_2;
          if (j + 1 < H && F < bufferIn[whichLine + (j + 1) * W]) {
            F         = bufferIn[whichLine + (j + 1) * W];
            indx[cnt] = whichLine + (j + 1) * W;
            DirH      = 0;
            DirV      = 1;
            // JOE lengthDir = 1;
          }
          if (whichLine + 1 < W && F < bufferIn[whichLine + 1 + (j) *W]) {
            F         = bufferIn[whichLine + 1 + (j) *W];
            indx[cnt] = whichLine + 1 + (j) *W;
            DirH      = 1;
            DirV      = 0;
            // JOE lengthDir = 1;
          }
          whichLine += DirH;
          j += DirV;
          cnt++;
        } while (whichLine <= W - 2 && j <= H - 2);
        rank_open(bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

      // 8 : direction bottom right to top left
      // printf ("direction 8: bottom right to top left\n");
      for (j = 1; j < H; j += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = W - 1 + j * W;
        whichLine = j;
        i         = W - 1;
        int cnt   = 1;

        do {
          F         = bufferIn[i - 1 + (whichLine - 1) * W];
          indx[cnt] = i - 1 + (whichLine - 1) * W;
          DirV      = -1;
          DirH      = -1;
          // JOE lengthDir = SQRT_2;
          if (whichLine - 1 >= 0 && F < bufferIn[i + (whichLine - 1) * W]) {
            F         = bufferIn[i + (whichLine - 1) * W];
            indx[cnt] = i + (whichLine - 1) * W;
            DirH      = 0;
            DirV      = -1;
            // JOE lengthDir = 1;
          }
          if (i - 1 >= 0 && F < bufferIn[i - 1 + (whichLine) *W]) {
            F         = bufferIn[i - 1 + (whichLine) *W];
            indx[cnt] = i - 1 + (whichLine) *W;
            DirH      = -1;
            DirV      = 0;
            // JOE lengthDir = 1;
          }
          i += DirH;
          whichLine += DirV;
          cnt++;
        } while (i >= 1 && whichLine >= 1);
        rank_open(bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

      for (i = 1; i < W; i += step) {
        // JOE wp        = 0;
        // JOE Length    = 0;
        // JOE stackSize = 0;
        indx[0]   = i + (H - 1) * W;
        whichLine = i;
        j         = H - 1;
        int cnt   = 1;

        do {
          F         = bufferIn[whichLine - 1 + (j - 1) * W];
          indx[cnt] = whichLine - 1 + (j - 1) * W;
          DirV      = -1;
          DirH      = -1;
          // JOE lengthDir = SQRT_2;
          if (j - 1 >= 0 && F < bufferIn[whichLine + (j - 1) * W]) {
            F         = bufferIn[whichLine + (j - 1) * W];
            indx[cnt] = whichLine + (j - 1) * W;
            DirH      = 0;
            DirV      = -1;
            // JOE lengthDir = 1;
          }
          if (whichLine - 1 >= 0 && F < bufferIn[whichLine - 1 + (j) *W]) {
            F         = bufferIn[whichLine - 1 + (j) *W];
            indx[cnt] = whichLine - 1 + (j) *W;
            DirH      = -1;
            DirV      = 0;
            // JOE lengthDir = 1;
          }
          whichLine += DirH;
          j += DirV;
          cnt++;
        } while (whichLine >= 1 && j >= 1);
        rank_open(bufferIn, indx, cnt, size, tolerance, bufferOut);
      }

      // JOE delete[] LineIdx;
      delete[] indx;

      return RES_OK;
    }
  };

  /*
   *
   *
   *
   */
  template <class T>
  RES_T parsimoniousPathOpening(Image<T> &imIn, int Size,
                                            int tolerance, int step,
                                            bool rebuild, Image<T> &imOut)
  {
    ParsimoniousPathOpeningClass<T> pOpen;

    RES_T res;

    res = pOpen.doIt(imIn, Size, tolerance, step, imOut);
    if (res != RES_OK) {
      ERR_MSG("Error doint path opening");
      return res;
    }

    /* Do rebuild */
    if (rebuild) {
      res = geoBuild(imOut, imIn, imOut);
      if (res != RES_OK)
        ERR_MSG("Error while rebuilding after Path Opening");
    }

    return res;
  }


} // namespace smil

#endif // __FAST_PIPATH_OPENING_HPP__
