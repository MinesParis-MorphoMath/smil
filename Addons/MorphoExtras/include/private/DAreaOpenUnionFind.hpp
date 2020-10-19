/*
 * __HEAD__
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
 *   Area Open using UnionFind Method
 *
 * History :
 *   - 15/10/2020 - by Jose-Marcio Martins da Cruz
 *     Porting from xxx
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_AREA_OPEN_UNION_FIND_HPP
#define _D_AREA_OPEN_UNION_FIND_HPP

#include <cmath>
#include <cstring>
#include <iomanip>

#include "Core/include/DCore.h"

using namespace std;

namespace smil
{
  /**
   * @addtogroup AddonMorphoExtrasAttrOpen      Attribute Open/Close
   *
   * @{
   */
  /** @cond */
  template <typename T>
  class UnionFindFunctions
  {
    //
    // P U B L I C
    //
  public:
    UnionFindFunctions()
    {
    }

    RES_T areaOpen(const Image<T> &imIn, size_t size, Image<T> &imOut)
    {
      _init(imIn);

      bufOut = (T *) imOut.getPixels();
      lambda = size;

      fill(imOut, T(0));
      ImageFreezer freeze(imOut);

      return _areaOpen(imIn, size, imOut);
    }

    //
    // P R I V A T E
    //
  private:
    //
    // F I E L D S
    //
    off_t width;
    off_t height;
    off_t depth;

    off_t lambda;

    Image<T> imIn;
    typename Image<T>::lineType bufIn;
    typename Image<T>::lineType bufOut;

    map<T, vector<off_t>> histoMap;
    vector<off_t> parent;

    bool debug;

    //
    // M E T H O D S
    //
    void mkHistogram(const Image<T> &im)
    {
      typename Image<T>::lineType pixels = im.getPixels();
      for (size_t i = 0; i < im.getPixelCount(); i++) {
        T val = pixels[i];

        size_t capacity = histoMap[val].capacity();
        if ((capacity - histoMap[val].size()) < 1024)
          histoMap[val].reserve(capacity + 1024);

        histoMap[val].push_back(i);
      }

      // just to debug...
      if (debug) {
        for (auto itk = histoMap.rbegin(); itk != histoMap.rend(); itk++) {
          cout << "Level \t" << int(itk->first) << "\t" << itk->second.size()
               << endl;
        }
        cout << endl;

        for (auto itk = histoMap.begin(); itk != histoMap.end(); itk++) {
          auto i = 0;
          cout << endl;
          cout << "* Histogram Level " << setw(4) << int(itk->first) << "\t"
               << itk->second.size() << endl;
          for (auto itv = itk->second.begin(); itv != itk->second.end();
               itv++, i++) {
            if (i % 16 == 0)
              cout << endl;
            cout << " " << setw(4) << *itv;
          }
          cout << endl;
        }
      }
    }

    void _init(const Image<T> &im)
    {
      imIn = im;

      width  = imIn.getWidth();
      height = imIn.getHeight();
      depth  = imIn.getDepth();

      parent.resize(imIn.getPixelCount(), 0);

      bufIn = (T *) imIn.getPixels();

      debug = false;
    }

    // Union Find functions
    void MakeSet(off_t x)
    {
      parent[x] = -1;
    }

    off_t FindRoot(off_t x)
    {
      if (parent[x] >= 0 && parent[x] != x) {
        parent[x] = FindRoot(parent[x]);
        return parent[x];
      } else
        return x;
    }

    bool Criterion(off_t x, off_t y)
    {
      return (bufIn[x] == bufIn[y] || -parent[x] < lambda);
    }

    void Union(off_t n, off_t p)
    {
      off_t r = FindRoot(n);

      if (r != p) {
        if (Criterion(r, p)) {
          parent[p] = parent[p] + parent[r];
          parent[r] = p;
        } else {
          parent[p] = -lambda;
        }
      }
    }

    // debug calls
    template <typename TD>
    void dumpVector(vector<TD> &b, string head = "")
    {
      cout << endl;
      cout << "=========================== " << head << endl;
      for (auto i = 0; i < height; i++) {
        cout << "Line " << setw(4) << i << " - ";
        for (auto j = 0; j < width; j++) {
          cout << " " << setw(4) << int(b[i * width + j]);
        }
        cout << endl;
      }
    }

    template <typename TI>
    void dumpImage(TI *b, string head = "")
    {
      cout << endl;
      cout << "=========================== " << head << endl;
      for (auto i = 0; i < height; i++) {
        cout << "Line " << setw(4) << i << " - ";
        for (auto j = 0; j < width; j++) {
          cout << " " << setw(4) << int(b[i * width + j]);
        }
        cout << endl;
      }
    }

    // area open
    RES_T _areaOpen(const Image<T> &imIn, size_t size, Image<T> &imOut)
    {
      bufOut = (T *) imOut.getPixels();
      lambda = size;

      mkHistogram(imIn);

      // Tarjan algorithm
      for (auto itk = histoMap.rbegin(); itk != histoMap.rend(); itk++) {
        for (auto itv = itk->second.begin(); itv != itk->second.end(); itv++) {
          off_t pix = *itv;
          off_t nbg;
          off_t dc;

          off_t x = pix % width;
          off_t y = (pix - x) / width;
          off_t z = (pix - x - y * width) / (width * height);

          MakeSet(pix);

          dc = -1;
          if (x + dc < 0 || x + dc >= width)
            continue;
          nbg = pix + dc;
          if ((bufIn[pix] < bufIn[nbg]) ||
              ((bufIn[pix] == bufIn[nbg]) && (nbg < pix)))
            Union(nbg, pix);

          dc = 1;
          if (x + dc < 0 || x + dc >= width)
            continue;
          nbg = pix + dc;
          if ((bufIn[pix] < bufIn[nbg]) ||
              ((bufIn[pix] == bufIn[nbg]) && (nbg < pix)))
            Union(nbg, pix);

          dc = -1;
          if (y + dc < 0 || y + dc >= height)
            continue;
          nbg = pix + dc * width;
          if ((bufIn[pix] < bufIn[nbg]) ||
              ((bufIn[pix] == bufIn[nbg]) && (nbg < pix)))
            Union(nbg, pix);

          dc = 1;
          if (y + dc < 0 || y + dc >= height)
            continue;
          nbg = pix + dc * width;
          if ((bufIn[pix] < bufIn[nbg]) ||
              ((bufIn[pix] == bufIn[nbg]) && (nbg < pix)))
            Union(nbg, pix);

          dc = -1;
          if (z + dc < 0 || z + dc >= depth)
            continue;
          nbg = pix + dc * width * height;
          if ((bufIn[pix] < bufIn[nbg]) ||
              ((bufIn[pix] == bufIn[nbg]) && (nbg < pix)))
            Union(nbg, pix);

          dc = 1;
          if (y + dc < 0 || y + dc >= depth)
            continue;
          nbg = pix + dc * width * height;
          if ((bufIn[pix] < bufIn[nbg]) ||
              ((bufIn[pix] == bufIn[nbg]) && (nbg < pix)))
            Union(nbg, pix);
        }
      }

      // Making output image
      for (auto itk = histoMap.begin(); itk != histoMap.end(); itk++) {
        for (auto itv = itk->second.rbegin(); itv != itk->second.rend();
             itv++) {
          off_t pix = *itv;

          if (parent[pix] >= 0)
            parent[pix] = parent[parent[pix]];
          else
            parent[pix] = bufIn[pix];

          bufOut[pix] = T(parent[pix]);
        }
      }
      return RES_OK;
    }
  };
  /** @endcond */
  /**
   * @}
   */

} // namespace smil

#endif // _D_AREA_OPEN_UNION_FIND_HPP
