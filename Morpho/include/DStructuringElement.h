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

#include <string>

namespace smil
{
  /**
   * @addtogroup StrElt
   *
   * @details Morphological Structuring Elements Definitions
   *
   * @{
   */
  enum seType {
    SE_Generic,
    SE_Hex,
    SE_Hex0,
    SE_Squ,
    SE_Squ0,
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
    /** Class constructor - generic structurant element
     * @param[in] s : size of the structinrg element
     */
    StrElt(UINT s = 1)
        : BaseObject("StrElt"), odd(false), seT(SE_Generic), size(s)
    {
      this->setName();
    }

    /** Class constructor - clone another structuring element
     *
     * @param[in] rhs : structuring element
     */
    StrElt(const StrElt &rhs) : BaseObject(rhs)
    {
      this->clone(rhs);
      this->setName();
    }

#ifndef SWIG
    /** @cond */
    /* Available only under C++ */
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
      this->setName();
    }
    /** @endcond */
#endif // SWIG

    /**
     * Class constructor
     *
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
      vector<UINT>::iterator it;
      for (it = indexList.begin(); it != indexList.end(); it++)
        addPoint(*it);
      this->setName();
    }

    /** @cond */
    ~StrElt()
    {
    }
    /** @endcond */

    /**
     * getPoint() - Get the coordinates (as a point) of the pixel of order @c i
     * in the structuring element
     *
     * @param[in] i : pixel index
     * @returns the coordinates of the pixel, relative to the StrElt center
     */
    IntPoint getPoint(const UINT i)
    {
      return points[i];
    }

    /**
     * getSize() - Get the size of the Structuring Element
     *
     * @returns the size of the structuring element
     */
    UINT getSize() const
    {
      return size;
    }

    /**
     * Clone a structuring element
     *
     * Clone a strunturing element to, eventually, create another one based on
     * it.
     *
     * @b Example:
     * @code{.py}
     * import smilPython as sp
     *
     * se = sp.VertSE()
     * se.addPoint(1,0)
     * print(se)
     * # print result :
     * Structuring Element
     * Type      : 5    VertSE
     * Size      : 1
     * Point Nbr : 4
     * #1: (0,0,0)
     * #2: (0,-1,0)
     * #3: (0,1,0)
     * #4: (1,0,0)
     * @endcode
     */
    StrElt &operator=(const StrElt &rhs);

    /**
     * clone() - Clone a structuring element
     *
     * @param[in] rhs : structuring element to be cloned
     * @returns a structuring element
     */
    void clone(const StrElt &rhs);

    //! List of neighbor points
    vector<IntPoint> points;

    /**
     * addPoint() - Add a point to the structurant element based on an index on
     * a grid.
     *
     * Index are defined as in the following drawings :
     * - Grids : @txtbold{Square} and @txtbold{Hexagonal}
     *   @images{grids}
     *
     * @param[in] index : index to predefined point coordinates, as above.
     *
     * @b Example:
     * @code{.py}
     * import smilPython as sp
     *
     * # Create a diagonal structuring element in a square grid,
     * diagSE_s = sp.StrElt(False)
     * diagSE_s.addPoint(0)
     * diagSE_s.addPoint(4)
     * diagSE_s.addPoint(8)
     *
     * # Create a diagonal structuring element in an hexagonal grid:
     * diagSE_h = sp.StrElt(True)
     * diagSE_h.addPoint(0)
     * diagSE_h.addPoint(3)
     * diagSE_h.addPoint(6)
     * @endcode
     */
    void addPoint(const UINT index);

    /**
     * addPoint() - Add a point to the structurant element given its coordinates
     *
     * @param[in] x, y, [z] : point coordinates
     */
    void addPoint(int x, int y, int z = 0);

    /**
     * addPoint() - Add a point to the structurant element given its coordinates
     * in a @txttype{IntPoint} data structure.
     *
     * @param[in] pt : a point to be added to the structuring element (itself)
     */
    void addPoint(const IntPoint &pt);

    /**
     * @b operator() -
     */
    const StrElt operator()(int s = 1) const;

    /**
     * homothety() - Build and return an homothetic SE with size @b s
     *
     * @param[in] s : size of the new structuring element
     * @returns a structuring element of size @b s
     */
    StrElt homothety(const UINT s) const;

    /**
     * transpose() - Return the opposite SE (symmetry with respect to 0)
     *
     * @returns a structuring element
     */
    StrElt transpose() const;

    /**
     * Return the SE with no center
     *
     * Remove the central point of the Structuring Element
     *
     * @return a structuring element
     */
    StrElt noCenter() const;

    /**
     * getType() - Get the type of the structuring element
     *
     * @return the content of the @b seT private field
     */
    virtual seType getType() const
    {
      return seT;
    }

    /** setName() - Set the name of the structuring element
     *
     * @param[in] name - the new name
     */
    void setName(string name)
    {
      this->name = name;
    }

    /** setName() - Set the name of the structuring element
     *
     * Set the structuring element based on the type field @txtbold{seT}
     */
    void setName()
    {
      std::map<seType, string> seNames = {
          {SE_Generic, "GenericSE"},
          {SE_Squ, "SquSE"},
          {SE_Squ0, "SquSE0"},
          {SE_Hex, "HexSE"},
          {SE_Hex0, "HexSE0"},
          {SE_Cross, "CrossSE"},
          {SE_Horiz, "HorizSE"},
          {SE_Vert, "VertSE"},
          {SE_Cube, "CubeSE"},
          {SE_Cross3D, "Cross3DSE"},
          {SE_Rhombicuboctahedron, "RhombicuboctahedronSE"},
      };

#if 0
      seNames[SE_Squ]                 = "SquSE";
      seNames[SE_Squ0]                = "SquSE0";
      seNames[SE_Hex]                 = "HexSE";
      seNames[SE_Hex0]                = "HexSE0";
      seNames[SE_Cross]               = "CrossSE";
      seNames[SE_Horiz]               = "HorizSE";
      seNames[SE_Vert]                = "VertSE";
      seNames[SE_Cube]                = "CubeSE";
      seNames[SE_Cross3D]             = "Cross3DSE";
      seNames[SE_Rhombicuboctahedron] = "RhombicuboctahedronSE";
      seNames[SE_Generic]             = "GenericSE";
#endif
      std::map<seType, string>::iterator it;
      it = seNames.find(seT);
      if (it != seNames.end())
        this->name = seNames[seT];
      else
        this->name = "Unknown";
    }

    /**
     * getName() - Get the name of the structurint element
     *
     * @returns the name of the structuring element (as a string)
     */
    string getName()
    {
      return name;
    }

    /**
    * printSelf() - Print the contents of the structuring element
    *
    * @note
    * In @txtbold{Python} this has the same effect than @txtbold{print(se)}
    */
    virtual void printSelf(ostream &os = std::cout, string indent = "") const;

    bool odd;
    seType seT;
    UINT size;
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
      className = "SquSE : StrElt";
      seT       = SE_Squ;
      size      = s;
      this->setName();
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
      seT       = SE_Squ0;
      odd       = false;
      size      = s;
      this->setName();
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
      className = "HexSE : StrElt";
      seT       = SE_Hex;
      size      = s;
      this->setName();
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
      seT       = SE_Hex0;
      size      = s;
      this->setName();
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
      className = "CrossSE : StrElt";
      seT       = SE_Cross;
      size      = s;
      this->setName();
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
      className = "HorizSE : StrElt";
      seT       = SE_Horiz;
      size      = s;
      this->setName();
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
      className = "VertSE : StrElt";
      seT       = SE_Vert;
      size      = s;
      this->setName();
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
      className = "CubeSE : StrElt";
      seT       = SE_Cube;
      odd       = false;
      this->setName();

      int zList[] = {0, -1, 1};
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
      className = "Cross3DSE : StrElt";
      seT       = SE_Cross3D;
      odd       = false;
      this->setName();

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
      className = "RhombicuboctahedronSE : StrElt";
      seT       = SE_Rhombicuboctahedron;
      odd       = false;
      this->setName();

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
  /** @cond */
  /* Only available inside C++ programs */
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
  /** @endcond */

#define DEFAULT_SE Morpho::getDefaultSE()

  /** @} */

} // namespace smil

#endif
