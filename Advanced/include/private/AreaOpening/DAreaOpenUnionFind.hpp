/*
 * __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2024, Centre de Morphologie Mathematique
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
#include <mutex>

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"

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
      se = DEFAULT_SE;
      // se = CrossSE();
    }

    RES_T areaOpen(const Image<T> &imIn, size_t size, Image<T> &imOut,
                   StrElt &se)
    {
      _init(imIn);

      this->se = se;

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
    off_t slice;

    off_t lambda;

    Image<T> imIn;
    typename Image<T>::lineType bufIn;
    typename Image<T>::lineType bufOut;

    StrElt se;

    std::map<T, std::vector<off_t>> histoMap;
    std::vector<off_t> parent;

    bool debug;

    //
    // M E T H O D S
    //
    void _init(const Image<T> &im)
    {
      imIn = im;

      width  = imIn.getWidth();
      height = imIn.getHeight();
      depth  = imIn.getDepth();
      slice  = height * width;

      parent.resize(imIn.getPixelCount(), 0);

      bufIn = (T *) imIn.getPixels();

      debug = false;
    }

    //
    // H I S T O G R A M    M A P
    //
    void mkHistogram(const Image<T> &im)
    {
      typename Image<T>::lineType pixels = im.getPixels();
      std::mutex mtx;

#ifdef USE_OPEN_MP
#pragma omp for
#endif
      for (size_t i = 0; i < im.getPixelCount(); i++) {
        T val = pixels[i];

        mtx.lock();
        size_t capacity = histoMap[val].capacity();
        if ((capacity - histoMap[val].size()) < 512)
          histoMap[val].reserve(capacity + 8192);

        histoMap[val].push_back(i);
        mtx.unlock();
      }

      if (debug)
        dumpHistogram();
    }

    //
    // U N I O N   F I N D
    //
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
          parent[p] += parent[r];
          parent[r] = p;
        } else {
          parent[p] = -lambda;
        }
      }
    }

    //
    // D E B U G  : D U M P   D A T A
    //

    // vector
    template <typename TD>
    void dumpVector(std::vector<TD> &b, std::string head = "")
    {
      std::cout << std::endl;
      std::cout << "=========================== " << head << std::endl;
      for (auto i = 0; i < height; i++) {
        std::cout << "Line " << std::setw(4) << i << " - ";
        for (auto j = 0; j < width; j++) {
          std::cout << " " << std::setw(4) << int(b[i * width + j]);
        }
        std::cout << std::endl;
      }
    }

    // image
    template <typename TI>
    void dumpImage(TI *b, std::string head = "")
    {
      std::cout << std::endl;
      std::cout << "=========================== " << head << std::endl;
      for (auto i = 0; i < height; i++) {
        std::cout << "Line " << std::setw(4) << i << " - ";
        for (auto j = 0; j < width; j++) {
          std::cout << " " << std::setw(4) << int(b[i * width + j]);
        }
        std::cout << std::endl;
      }
    }

    // histogram
    void dumpHistogram()
    {
      for (auto itk = histoMap.rbegin(); itk != histoMap.rend(); itk++) {
        std::cout << "Level \t" << int(itk->first) << "\t" << itk->second.size()
             << std::endl;
      }
      std::cout << std::endl;

      for (auto itk = histoMap.begin(); itk != histoMap.end(); itk++) {
        auto i = 0;
        std::cout << std::endl;
        std::cout << "* Histogram Level " << std::setw(4) << int(itk->first) << "\t"
             << itk->second.size() << std::endl;
        for (auto itv = itk->second.begin(); itv != itk->second.end();
             itv++, i++) {
          if (i % 16 == 0)
            std::cout << std::endl;
          std::cout << " " << std::setw(4) << *itv;
        }
        std::cout << std::endl;
      }
    }

    //
    // L I T T L E   T O O L S
    //
    void _pixel2coords(off_t pixel, off_t &x, off_t &y, off_t &z)
    {
      x = pixel % width;
      if (depth > 1) {
        y = (pixel % slice) / width;
        z = pixel / slice;
      } else {
        y = pixel / width;
        z = 0;
      }
    }

    off_t _coords2pixel(off_t &x, off_t &y, off_t &z)
    {
      if (depth > 1)
        return x + y * width + z * slice;
      else
        return x + y * width;
    }

    bool _pixelInWindow(off_t x, off_t y, off_t z)
    {
      if (x < 0 || x >= width)
        return false;
      if (y < 0 || y >= height)
        return false;
      if (z < 0 || z >= depth)
        return false;
      return true;
    }

    //
    // A R E A   O P E N
    //
    RES_T _areaOpen(const Image<T> &imIn, size_t size, Image<T> &imOut)
    {
      bufOut = (T *) imOut.getPixels();
      lambda = size;

      // build histogram as a map
      mkHistogram(imIn);

      // remove central pixel on the structuring element
      se = se.noCenter();

      // Tarjan algorithm
      for (auto itk = histoMap.rbegin(); itk != histoMap.rend(); itk++) {
        for (auto itv = itk->second.begin(); itv != itk->second.end(); itv++) {
          off_t pix = *itv;
          off_t nbg;

          off_t x, y, z;
          _pixel2coords(pix, x, y, z);

          MakeSet(pix);

          for (auto its = se.points.begin(); its != se.points.end(); its++) {
            off_t dx = its->x;
            off_t dy = its->y;
            off_t dz = its->z;

            if (!_pixelInWindow(x + dx, y + dy, z + dz))
              continue;
            nbg = pix + _coords2pixel(dx, dy, dz);

            if ((bufIn[pix] < bufIn[nbg]) ||
                ((bufIn[pix] == bufIn[nbg]) && (nbg < pix)))
              Union(nbg, pix);
          }
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
