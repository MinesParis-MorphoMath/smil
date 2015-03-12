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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _DCOMMON_H
#define _DCOMMON_H


#include <string.h>
#include <memory>
#include <limits>
#include <vector>
#include <map>
#include <math.h>
#include <stdarg.h>


#include "private/DTypes.hpp"

using namespace std;

namespace smil
{
    #define VERBOSE 1

    #if VERBOSE > 1
    #define MESSAGE(msg) cout << msg <<  endl;
    #else // VERBOSE 
    #define MESSAGE(msg)
    #endif // VERBOSE 



    #define INLINE inline




    #define SMART_POINTER(T) boost::shared_ptr< T >
    #define SMART_IMAGE(T) SMART_POINTER( D_Image< T > )

    #define D_DEFAULT_IMAGE_WIDTH 512
    #define D_DEFAULT_IMAGE_HEIGHT 512
    #define D_DEFAULT_IMAGE_DEPTH 1

    #define D_DEFAULT_OUT_PIXEL_VAL 0

    #ifndef PI
    #define PI 3.141592653589793
    #endif // PI

    #ifndef MIN
    #define MIN(a, b) (a < b ? a : b);
    #endif // MIN
    #ifndef MAX
    #define MAX(a, b) (a > b ? a : b);
    #endif // MAX

    template <class T>
    struct Point
    {
      T x;
      T y;
      T z;
      Point() : x(0), y(0), z(0) {}
      Point(const Point &pt) : x(pt.x), y(pt.y), z(pt.z) {}
      Point(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
      Point(T _x, T _y) : x(_x), y(_y), z(0) {}
      bool operator == (const Point &p2)
      {
          return (x==p2.x && y==p2.y && z==p2.z);
      }
    };

    typedef Point<double> DoublePoint;
    typedef Point<int> IntPoint;
    
    typedef vector<double> Vector_double;
    typedef vector<Vector_double> Matrix_double;
    typedef vector<UINT> Vector_UINT;
    
    struct Rectangle
    {
        UINT x0, y0;
        UINT xSzie, ySize;
    };

    struct Box
    {
        UINT x0, y0, z0;
        UINT x1, y1, z1;
        Box()
        {
            x0 = x1 = y0 = y1 = z0 = z1 = 0;
        }
        Box(const Box &rhs)
        {
            x0 = rhs.x0;
            x1 = rhs.x1;
            y0 = rhs.y0;
            y1 = rhs.y1;
            z0 = rhs.z0;
            z1 = rhs.z1;
        }
        UINT getWidth() const { return x1-x0+1; }
        UINT getHeight() const { return y1-y0+1; }
        UINT getDepth() const { return z1-z0+1; }
    };

    
    // Misc Macros

    #ifdef _MSC_VER

    #define __FUNC__ __FUNCTION__

    // Work-around to MSVC __VA_ARGS__ expanded as a single argument, instead of being broken down to multiple ones
    #define EXPAND( ... ) __VA_ARGS__

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

    #define _xPP_NARGS_IMPL(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,N,...) N
    #define PP_NARGS(...) \
        EXPAND(_xPP_NARGS_IMPL(__VA_ARGS__,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0))

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

    #define _xPP_NARGS_IMPL(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,N,...) N
    #define PP_NARGS(...) \
        _xPP_NARGS_IMPL(__VA_ARGS__,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)

    #endif // _MSC_VER

} // namespace smil

#endif // _DCOMMON_H

