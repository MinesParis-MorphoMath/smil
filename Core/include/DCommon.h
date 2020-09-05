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

#ifndef _DCOMMON_H
#define _DCOMMON_H

#include <cstring>
#include <memory>
#include <limits>
#include <vector>
#include <map>
#include <cmath>
#include <cstdarg>

#include "private/DTypes.hpp"

using namespace std;

namespace smil
{
  /**
  * @addtogroup  CoreExtraTypes
  *
  * @brief   Some useful data types when handling images
  *
  * @{
  */
  
#define VERBOSE 1

#if VERBOSE > 1
#define MESSAGE(msg) cout << msg << endl;
#else // VERBOSE
#define MESSAGE(msg)
#endif // VERBOSE

#if defined __GNUC__ || defined __clang__
#define SMIL_UNUSED __attribute__((__unused__))
#else // MSVC et al.
#define SMIL_UNUSED
#endif

#define INLINE inline

  // Generate template specializations (or instanciations?) for
  // ImageHandler subclasses (in IO/DImageIO_{BMP,JPG,PBM,PNG,TIFF})
#define IMAGEFILEHANDLER_TEMP_SPEC(FORMAT, PIXELTYPE)                          \
  template <>                                                                  \
  class FORMAT##ImageFileHandler<PIXELTYPE>                                    \
      : public ImageFileHandler<PIXELTYPE>                                     \
  {                                                                            \
  public:                                                                      \
    FORMAT##ImageFileHandler() : ImageFileHandler<PIXELTYPE>(#FORMAT)          \
    {                                                                          \
    }                                                                          \
    RES_T read(const char *filename, Image<PIXELTYPE> &image);                 \
    RES_T write(const Image<PIXELTYPE> &image, const char *filename);          \
  };

#define SMART_POINTER(T) boost::shared_ptr<T>
#define SMART_IMAGE(T) SMART_POINTER(D_Image<T>)

#define D_DEFAULT_IMAGE_WIDTH 512
#define D_DEFAULT_IMAGE_HEIGHT 512
#define D_DEFAULT_IMAGE_DEPTH 1

#define D_DEFAULT_OUT_PIXEL_VAL 0

#ifndef PI
#define PI 3.141592653589793
#endif // PI

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif // MIN
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif // MAX
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif // ABS

#ifndef SWIG
  struct map_comp_value_less {
    template <typename Lhs, typename Rhs>
    bool operator()(const Lhs &lhs, const Rhs &rhs) const
    {
      return lhs.second < rhs.second;
    }
  };
#endif // SWIG

  /**
    * Struct Point
    */
  template <class T> struct Point {
    T x;
    T y;
    T z;

    /** Contructor - an empty point
    */
    Point() : x(0), y(0), z(0)
    {
    }

    /** Constructor from another point
    */
    Point(const Point &pt) : x(pt.x), y(pt.y), z(pt.z)
    {
    }

    /** @b 3D Point Constructor
    */
    Point(T _x, T _y, T _z) : x(_x), y(_y), z(_z)
    {
    }

    /** @b 2D Point Constructor
    */
    Point(T _x, T _y) : x(_x), y(_y), z(0)
    {
    }

    /** operator== - comparison
    */
    bool operator==(const Point &p2)
    {
      return (x == p2.x && y == p2.y && z == p2.z);
    }
  };

  /** DoublePoint
  *
  * Point coordinates defined as @b double values
  */
  typedef Point<double> DoublePoint;

  /** IntPoint
  *
  * Point coordinates defined as @b int values
  */
  typedef Point<int> IntPoint;

  /** UintPoint
  *
  * Point coordinates defined as @b UINT (unsigned int) values
  */
  typedef Point<UINT> UintPoint;

  /** Vector_double
  *
  * A vector of @b double values
  */
  typedef vector<double> Vector_double;

  /** Matrix_double
  *
  * A Matrix of @b double values implemented as a vector of vectors
  */
  typedef vector<Vector_double> Matrix_double;

  /** Vector_UINT
  *
  * A vector of @b UINT (unsigned int) values
  */
  typedef vector<UINT> Vector_UINT;

  /** Vector_size_t
  *
  * A vector of @b size_t values (natural - non negative values)
  */
  typedef vector<size_t> Vector_size_t;

  /** Vector_off_t
  *
  * A vector of @b off_t values (integer - positive and negative values)
  */
  typedef vector<off_t> Vector_off_t;

  /**
   * Rectangle
   */
  struct Rectangle {
    UINT x0, y0;
    UINT xSize, ySize;
  };

  /** Box
  */
  struct Box {
    UINT x0, y0, z0;
    UINT x1, y1, z1;

    /** Box constructor - build an empty Box structure
    */
    Box()
    {
      x0 = x1 = y0 = y1 = z0 = z1 = 0;
    }

    /** Box constructor - build a Box copying data from another Box
    */
    Box(const Box &rhs)
    {
      x0 = rhs.x0;
      x1 = rhs.x1;
      y0 = rhs.y0;
      y1 = rhs.y1;
      z0 = rhs.z0;
      z1 = rhs.z1;
    }
    
    /** getWidth() - Get the box width
    * @returns box width
    */
    UINT getWidth() const
    {
      return x1 - x0 + 1;
    }

    /** getHeight() - Get the box width
    * @returns box height
    */
    UINT getHeight() const
    {
      return y1 - y0 + 1;
    }

    /** getDepth() - Get the box depth
    * @returns box depth
    */
    UINT getDepth() const
    {
      return z1 - z0 + 1;
    }
  };

  /** OffsetPoint 
  *
  * A structure with offset and point coordinates and some methods to handle
  * them.
  *
  * Shall be initialized with the dimensions of the image, in order to be able
  * to convert from @b Point to @b Offset and vice-versa, and to check if a
  * point is inside the image bounds.
  */
  struct OffsetPoint {
    off_t x, y, z;
    off_t o;
    off_t w, h, d;

    /** OffsetPoint - constructor
    * @details Build the data structure based on the image bounds
    * @param[in] Sz : vector with the three image bounds
    */
    OffsetPoint(size_t Sz[3])
    {
      w = Sz[0];
      h = Sz[1];
      d = Sz[2];
      x = y = z = o = 0;
    }

    /** OffsetPoint - constructor
    *
    * @details Build the data structure copying data from another OffsetPoint
    * data
    * @param[in] offset :
    */
    OffsetPoint(const OffsetPoint &offset)
    {
      w = offset.w;
      h = offset.h;
      d = offset.d;
      x = offset.x;
      y = offset.y;
      z = offset.z;
      o = offset.o;
    }

    /** OffsetPoint - constructor
    *
    * @param[in] w, h, d : image bounds
    */
    OffsetPoint(size_t w, size_t h, size_t d = 1)
    {
      this->w = w;
      this->h = h;
      this->d = d;
      x = y = z = o = 0;
    }

    /** setCoords() - set coordinates and adapt offset
    * @param[in] x, y, z :
    */
    void setCoords(off_t x, off_t y, off_t z = 0)
    {
      this->x = x;
      this->y = y;
      this->z = z;
      this->o = x + y * w + z * w * h;
    }

    /** setOffset() - set offset and adapt coordinates
    * @param[in] offset :
    */
    void setOffset(off_t offset)
    {
      this->o = offset;

      this->x = offset % w;
      offset  = (offset - this->x) / w;
      this->y = offset % h;
      this->z = (offset - this->y) / h;
    }

    /** getPoint() - get coordinates as a point
    * @returns The coordinates of the structure as a point
    */
    IntPoint getPoint()
    {
      IntPoint pt(x, y, z);
      return pt;
    }

    /** getOffset() - get coordinates as an offset
    * @returns The coordinates of the structure as an offset
    */
    off_t getOffset()
    {
      return o;
    }

    /** shift() - move the point by some displacements
    * @param[in] dx, dy, [dz] : amount to shift the offset structure
    */
    void shift(off_t dx, off_t dy, off_t dz = 0)
    {
      x += dx;
      y += dy;
      z += dz;
      this->o = x + y * w + z * w * h;
    }

    /** shift() - move the point by some displacements given by a point
    * @param[in] p : 
    */ 
    void shift(IntPoint p)
    {
      x += p.x;
      y += p.y;
      z += p.z;
      this->o = x + y * w + z * w * h;
    }

    /** inImage() - check if point coordinates is inside image bounds
    * @param[in] x, y, [z] :
    * @returns @b True if the three coordinates are inside image bounds, 
    * @b False otherwise
    */
    bool inImage(off_t x, off_t y, off_t z)
    {
      return (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d);
    }

    /** inImage() - check if point coordinates is inside image bounds
    * @returns @b True if the three coordinates are inside image bounds, 
    * @b False otherwise
    */
    bool inImage()
    {
      return (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d);
    }
  };


  // Misc Macros
  /** @cond */
#ifdef _MSC_VER

#define __FUNC__ __FUNCTION__

// Work-around to MSVC __VA_ARGS__ expanded as a single argument, instead of
// being broken down to multiple ones
#define EXPAND(...) __VA_ARGS__

#define _GET_1ST_ARG(arg1, ...) arg1
#define _GET_2ND_ARG(arg1, arg2, ...) arg2
#define _GET_3RD_ARG(arg1, arg2, arg3, ...) arg3
#define _GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4
#define _GET_5TH_ARG(arg1, arg2, arg3, arg4, arg5, ...) arg5
#define _GET_6TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, ...) arg6
#define _GET_7TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, arg7, ...) arg7
#define _GET_8TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, ...) arg8
#define GET_1ST_ARG(...) EXPAND(_GET_1ST_ARG(__VA_ARGS__))
#define GET_2ND_ARG(...) EXPAND(_GET_2ND_ARG(__VA_ARGS__))
#define GET_3RD_ARG(...) EXPAND(_GET_3RD_ARG(__VA_ARGS__))
#define GET_4TH_ARG(...) EXPAND(_GET_4TH_ARG(__VA_ARGS__))
#define GET_5TH_ARG(...) EXPAND(_GET_5TH_ARG(__VA_ARGS__))
#define GET_6TH_ARG(...) EXPAND(_GET_6TH_ARG(__VA_ARGS__))
#define GET_7TH_ARG(...) EXPAND(_GET_7TH_ARG(__VA_ARGS__))
#define GET_8TH_ARG(...) EXPAND(_GET_8TH_ARG(__VA_ARGS__))

#define _xPP_NARGS_IMPL(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,     \
                        x13, x14, x15, N, ...)                                 \
  N
#define PP_NARGS(...)                                                          \
  EXPAND(_xPP_NARGS_IMPL(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,   \
                         4, 3, 2, 1, 0))

#else // _MSC_VER

#define __FUNC__ __func__

#define GET_1ST_ARG(arg1, ...) arg1
#define GET_2ND_ARG(arg1, arg2, ...) arg2
#define GET_3RD_ARG(arg1, arg2, arg3, ...) arg3
#define GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4
#define GET_5TH_ARG(arg1, arg2, arg3, arg4, arg5, ...) arg5
#define GET_6TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, ...) arg6
#define GET_7TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, arg7, ...) arg7
#define GET_8TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, ...) arg8

#define _xPP_NARGS_IMPL(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,     \
                        x13, x14, x15, N, ...)                                 \
  N
#define PP_NARGS(...)                                                          \
  _xPP_NARGS_IMPL(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, \
                  1, 0)

#endif // _MSC_VER
  /** @endcond */

  /** @} */
} // namespace smil

#endif // _DCOMMON_H
