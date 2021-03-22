/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2021, Centre de Morphologie Mathematique
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
 *     First port from MorphM
 *   - 10/02/2019 - by Jose-Marcio
 *     Integrated into Smil Addon "Path Opening" with some cosmetics
 *   - 08/10/2020 - Jose-Marcio
 *     Review + Convert to Functor, Improve C++ style, clean-up, modify
 *     functions to handle images other than just UINT8, ...
 *
 * __HEAD__ - Stop here !
 */

#ifndef __WINDOW_POINTS_HPP__
#define __WINDOW_POINTS_HPP__

#include <cmath>
#include <cstring>
#include "Core/include/DCore.h"

using namespace std;

namespace smil
{
  /*
   * @ingroup Addons
   * @addtogroup AddonMorphoExtras
   */
  /** @cond */
  /*
   * @ingroup AddonMorphoExtras
   * @defgroup AddonMorphoExtrasGeodesic Geodesic Extra Tools
   *
   * @{
   */
  
  //
  // ######  #    #  #    #   ####
  // #       #    #  ##   #  #    #
  // #####   #    #  # #  #  #
  // #       #    #  #  # #  #
  // #       #    #  #   ##  #    #
  // #        ####   #    #   ####
  //
  class WindowPoints
  {
  public:
    WindowPoints(off_t width, off_t height, off_t depth = 1)
        : width(width), height(height), depth(depth)
    {
    }

    off_t index(off_t x, off_t y, off_t z = 0)
    {
      return x + (y + z * height) * width;
    }

    void coords(off_t p, off_t &x, off_t &y, off_t &z)
    {
      if (p < 0)
        ERR_MSG("Invalid negative index value is negative");
      x = p % width;
      p = (p - x) / width;
      y = p % height;
      z = (p - y) / height;
      if (z > depth)
        ERR_MSG("Invalid slice index greater than image depth");
    }

    void coords(off_t p, off_t &x, off_t &y)
    {
      off_t z;
      coords(p, x, y, z);
    }

    off_t translate(off_t p, off_t dx, off_t dy, off_t dz)
    {
      off_t x, y, z;
      coords(p, x, y, z);
      x = max(off_t(0), min(x + dx, width - 1));
      y = max(off_t(0), min(y + dy, height - 1));
      z = max(off_t(0), min(z + dz, depth - 1));
      return index(x, y, z);
    }

    bool inWindow(off_t x, off_t y, off_t z)
    {
      if (x < 0 || x >= width)
        return false;
      if (y < 0 || y >= height)
        return false;
      if (z < 0 || z >= depth)
        return false;
      return true;
    }

    bool inWindow(off_t p)
    {
      off_t x, y, z;
      coords(p, x, y, z);
      return inWindow(x, y, z);
    }

  private:
    off_t width;
    off_t height;
    off_t depth;
  };
  /** @endcond */
  /* @} */

} // namespace smil
#endif
