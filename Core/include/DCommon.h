/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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

    #ifdef _MSC_VER

    #ifdef smilCore_EXPORTS
	    #define _DCORE __declspec(dllexport)
    //	#pragma message(" - Exporting smilCore")
    #else
	    #define _DCORE __declspec(dllimport)
    //	#pragma message(" - Importing smilCore")
    #endif

    #ifdef smilBase_EXPORTS
    #define _DBASE __declspec(dllexport)
    #else
    #define _DBASE __declspec(dllimport)
    #endif

    #ifdef smilIO_EXPORTS
    #define _DIO __declspec(dllexport)
    #else
    #define _DIO __declspec(dllimport)
    #endif

    #ifdef smilGui_EXPORTS
    #define _DGUI __declspec(dllexport)
    #else
    #define _DGUI __declspec(dllimport)
    #endif

    #ifdef smilMorpho_EXPORTS
    #define _DMORPHO __declspec(dllexport)
    #else
    #define _DMORPHO __declspec(dllimport)
    #endif

    #else // _MSC_VER

    #define _DCORE
    #define _DBASE
    #define _DIO
    #define _DGUI
    #define _DMORPHO

    #endif // _MSC_VER



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
    #define MIN(a, b) a < b ? a : b;
    #endif // MIN
    #ifndef MAX
    #define MAX(a, b) a > b ? a : b;
    #endif // MAX

    template <class T>
    struct Point
    {
      T x;
      T y;
      T z;
      Point() : x(0), y(0), z(0) {}
      Point(const Point<T> &pt) : x(pt.x), y(pt.y), z(pt.z) {}
      Point(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
      bool operator == (const Point<T> &p2)
      {
	  return (x==p2.x && y==p2.y && z==p2.z);
      }
    };

    typedef Point<int> IntPoint;
    typedef Point<UINT8> UCPoint;
    typedef Point<double> DoublePoint;

    
    inline double round(double r) 
    {
	return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
    }

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

