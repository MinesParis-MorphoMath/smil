#ifndef _DCOMMON_H
#define _DCOMMON_H


#include <string.h>
#include <memory>
#include <limits>
#include <vector>
#include <stdarg.h>

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

