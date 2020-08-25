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

#ifndef _STR_ELT_HPP
#define _STR_ELT_HPP

#include "Core/include/DCommon.h"
#include "Core/include/DBaseObject.h"

namespace smil
{
  /*
   * @defgroup StrElt Structuring Elements
   * @ingroup Morpho
   *
   * @details In mathematical morphology, a structuring element is a shape, used
   * to probe or interact with a given image, with the purpose of drawing
   * conclusions on how this shape fits or misses the shapes in the image. It is
   * typically used in morphological operations, such as dilation, erosion,
   * opening, and closing, as well as the hit-or-miss transform.
   *
   * @see 
   * <a href="https://en.wikipedia.org/wiki/Structuring_element">Structuring
   * Element</a>
   *
   * @{
   */

  /**
   * @ingroup StrElt
   * @{
   */
  enum seType {
    SE_Generic,
    SE_Hex,
    SE_Squ,
    SE_Cross,
    SE_Horiz,
    SE_Vert,
    SE_Cube,
    SE_Cross3D,
    SE_Rhombicuboctahedron
  };

  /**
   * Base structuring element
   */
  class StrElt : public BaseObject
  {
  public:
    StrElt(UINT s = 1)
        : BaseObject("StrElt"), odd(false), seT(SE_Generic), size(s)
    {
    }

    StrElt(const StrElt &rhs) : BaseObject(rhs)
    {
      this->clone(rhs);
    }

#ifndef SWIG
    StrElt(bool oddSE, UINT nbrPts, ...)
        : BaseObject("StrElt"), odd(oddSE), seT(SE_Generic), size(1)
    {
      UINT index;
      va_list vl;
      va_start(vl, nbrPts);

      for (UINT i = 0; i < nbrPts; i++) {
        index = va_arg(vl, UINT);
        addPoint(index);
      }
    }
#endif // SWIG

    /**
     * Construct a structuring element with points defined by their indexes.
     * @param oddSE Specify if we want to use an hexagonal grid (true) or a
     * square grid (false)
     * @param indexList The list of point indexes
     *
     * The index values are defined for each grid type as follow:
     * @images{se_indexes}
     *
     * @b Example:
     * @code{.py}
     * # Create a diagonal SE with the two points (0,0) and (1,1),
     * # on the square grid:
     * diagSE_s = StrElt(False, (0,8))
     * # on the hexagonal grid:
     * diagSE_h = StrElt(True, (0,6))
     * @endcode
     */
    StrElt(bool oddSE, vector<UINT> indexList)
        : BaseObject("StrElt"), odd(oddSE), seT(SE_Generic), size(1)
    {
      for (vector<UINT>::iterator it = indexList.begin(); it != indexList.end();
           it++)
        addPoint(*it);
    }

    ~StrElt()
    {
    }

    IntPoint getPoint(const UINT i)
    {
      return points[i];
    }
    UINT getSize() const
    {
      return size;
    }

    StrElt &operator=(const StrElt &rhs);
    void clone(const StrElt &rhs);

    //! List of neighbor points
    vector<IntPoint> points;

    void addPoint(const UINT index);
    void addPoint(int x, int y, int z = 0);
    void addPoint(const IntPoint &pt);
    const StrElt operator()(int s = 1) const;

    //! Construct and return an homothetic SE with size s
    StrElt homothety(const UINT s) const;

    //! Return the opposite SE (symmetry with respect to 0)
    StrElt transpose() const;

    //! Return the SE with no center
    StrElt noCenter() const;

    bool odd;
    seType seT;
    UINT size;
    virtual seType getType() const
    {
      return seT;
    }

    virtual void printSelf(ostream &os = std::cout, string indent = "") const;
  };

  inline void operator<<(ostream &os, StrElt &se)
  {
    se.printSelf(os);
  }

  /**
   * Square structuring element.
   *
   * Points :
   * @images{squ_se}
   *
   */
  class SquSE : public StrElt
  {
  public:
    SquSE(UINT s = 1) : StrElt(false, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    {
      className = "SquSE";
      seT       = SE_Squ;
      size      = s;
    }
  };

  /**
   * Square structuring element without center point.
   *
   * Points :
   * @images{squ_se0}
   *
   */

  class SquSE0 : public StrElt
  {
  public:
    typedef StrElt parentClass;
    SquSE0(UINT s = 1) : StrElt(false, 8, 1, 2, 3, 4, 5, 6, 7, 8)
    {
      className = "SquSE0";
      odd       = false;
      size      = s;
    }
  };

  /**
   * Hexagonal structuring element.
   *
   * Points :
   * @images{hex_se}
   *
   */
  class HexSE : public StrElt
  {
  public:
    HexSE(UINT s = 1) : StrElt(true, 7, 0, 1, 2, 3, 4, 5, 6)
    {
      className = "HexSE";
      seT       = SE_Hex;
      size      = s;
    }
  };

  /**
   * Hexagonal structuring element without center point.
   *
   * Points :
   * @images{hex_se0}
   *
   */

  class HexSE0 : public StrElt
  {
  public:
    HexSE0(UINT s = 1) : StrElt(true, 6, 1, 2, 3, 4, 5, 6)
    {
      className = "HexSE0";
      size      = s;
    }
  };

  /**
   * Cross structuring element.
   *
   * Points :
   * @images{cross_se}
   *
   */

  class CrossSE : public StrElt
  {
  public:
    CrossSE(UINT s = 1) : StrElt(false, 5, 0, 1, 5, 3, 7)
    {
      className = "CrossSE";
      seT       = SE_Cross;
      size      = s;
    }
  };

  /**
   * Horizontal segment structuring element.
   *
   * Points :
   * @images{horiz_se}
   *
   */

  class HorizSE : public StrElt
  {
  public:
    HorizSE(UINT s = 1) : StrElt(false, 3, 0, 1, 5)
    {
      className = "HorizSE";
      seT       = SE_Horiz;
      size      = s;
    }
  };

  /**
   * Vertical segment structuring element.
   *
   * Points :
   * @images{vert_se}
   *
   */

  class VertSE : public StrElt
  {
  public:
    VertSE(UINT s = 1) : StrElt(false, 3, 0, 3, 7)
    {
      className = "VertSE";
      seT       = SE_Vert;
      size      = s;
    }
  };

  /**
   * 3D Cubic structuring element (26 neighbors).
   *
   * Points :
   * @images{cube_se}
   *
   */
  class CubeSE : public StrElt
  {
  public:
    CubeSE(UINT s = 1) : StrElt(s)
    {
      this->className = "CubeSE";
      this->seT       = SE_Cube;
      odd             = false;
      int zList[]     = {0, -1, 1};
      for (int i = 0; i < 3; i++) {
        int z = zList[i];
        addPoint(0, 0, z);
        addPoint(1, 0, z);
        addPoint(1, -1, z);
        addPoint(0, -1, z);
        addPoint(-1, -1, z);
        addPoint(-1, 0, z);
        addPoint(-1, 1, z);
        addPoint(0, 1, z);
        addPoint(1, 1, z);
      }
    }
  };

  /**
   * 3D Cross structuring element (6 neighbors).
   *
   * Points :
   * @images{cross3d_se}
   *
   */
  class Cross3DSE : public StrElt
  {
  public:
    Cross3DSE(UINT s = 1) : StrElt(s)
    {
      className = "Cross3DSE";
      seT       = SE_Cross3D;
      odd       = false;
      addPoint(0, 0, 0);
      addPoint(1, 0, 0);
      addPoint(0, -1, 0);
      addPoint(-1, 0, 0);
      addPoint(0, 1, 0);
      addPoint(0, 0, -1);
      addPoint(0, 0, 1);
    }
  };

  /**
   * Rhombicuboctahedron struturing element (80 neighbors).
   * Points :
   * @images{rhombicuboctaedron_se}
   *
   */
  class RhombicuboctahedronSE : public StrElt
  {
  public:
    RhombicuboctahedronSE(UINT s = 1) : StrElt(s)
    {
      className = "RhombicuboctahedronSE";
      seT       = SE_Rhombicuboctahedron;
      odd       = false;

      int x, y, z;

      addPoint(0, 0, 0);
      for (x = -2; x <= 2; x++)
        for (y = -1; y <= 1; y++)
          for (z = -1; z <= 1; z++)
            addPoint(x, y, z);
      for (x = -1; x <= 1; x++)
        for (y = -2; y <= 2; y++)
          for (z = -1; z <= 1; z++)
            addPoint(x, y, z);
      for (x = -1; x <= 1; x++)
        for (y = -1; y <= 1; y++)
          for (z = -2; z <= 2; z++)
            addPoint(x, y, z);
    }
  };

  // Shortcuts
  inline HexSE hSE(UINT s = 1)
  {
    return HexSE(s);
  }
  inline HexSE0 hSE0(UINT s = 1)
  {
    return HexSE0(s);
  }
  inline SquSE sSE(UINT s = 1)
  {
    return SquSE(s);
  }
  inline SquSE0 sSE0(UINT s = 1)
  {
    return SquSE0(s);
  }
  inline CrossSE cSE(UINT s = 1)
  {
    return CrossSE(s);
  }
  inline CubeSE cbSE(UINT s = 1)
  {
    return CubeSE(s);
  }
  inline RhombicuboctahedronSE rcoSE(UINT s = 1)
  {
    return RhombicuboctahedronSE(s);
  }

#define DEFAULT_SE Morpho::getDefaultSE()

  /** @} */

} // namespace smil

#endif
