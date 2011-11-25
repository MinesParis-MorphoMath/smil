/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


#ifndef _DCOMMON_H
#define _DCOMMON_H


#include <string.h>
#include <memory>
#include <limits>
#include <vector>
#include <stdarg.h>


#include "DTypes.hpp"
// #include "auto_ptr.h"
// #include <boost/smart_ptr.hpp>

using namespace std;


#define VERBOSE 1

#if VERBOSE > 1
#define MESSAGE(msg) cout << msg <<  endl;
#else // VERBOSE 
#define MESSAGE(msg)
#endif // VERBOSE 



#define INLINE inline

#ifdef _MSC_VER
#ifdef smilCore_EXPORTS
// the dll exports
#define _SMIL __declspec(dllexport)
#else // smilCore_EXPORTS
// the exe imports
#define _SMIL __declspec(dllimport)
#endif // smilCore_EXPORTS
#else // _MSC_VER
#define _SMIL
#endif // _MSC_VER



#define SMART_POINTER(T) boost::shared_ptr< T >
#define SMART_IMAGE(T) SMART_POINTER( D_Image< T > )

#define D_DEFAULT_IMAGE_WIDTH 512
#define D_DEFAULT_IMAGE_HEIGHT 512
#define D_DEFAULT_IMAGE_DEPTH 1

#define D_DEFAULT_OUT_PIXEL_VAL 0



#if defined(__MINGW32__)

#if __GNUC__ < 4
#include <malloc.h>
#define MALLOC(ptr,size, align)  ptr = __mingw_aligned_malloc(size,align)
#define FREE(p)                  __mingw_aligned_free(p)
#else
#include <mm_malloc.h>
#define MALLOC(ptr,size,align)   ptr = _mm_malloc(size,align)
#define FREE(p)                  _mm_free(p)
#endif

#elif defined(_MSC_VER)

#include <malloc.h>
#define MALLOC(ptr,size,align)   ptr = _aligned_malloc(size,align)
#define FREE(p)                  _aligned_free(p)

#else

#include <mm_malloc.h>
#define MALLOC(ptr,size,align)   ptr = _mm_malloc(size,align)
#define FREE(p)                  _mm_free(p)

#endif



#endif // _DCOMMON_H

