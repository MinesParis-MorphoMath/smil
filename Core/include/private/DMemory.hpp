/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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


#ifndef _DMEMORY_HPP
#define _DMEMORY_HPP

#include <math.h>
#include <string>
#include <limits>

#include <stdio.h>

#if (defined(__ICL) || defined(__ICC))
  #include <fvec.h>
  inline void *aligned_malloc (size_t size, size_t align=16) { return _mm_malloc(size,align); }
  inline void  aligned_free   (void *p)                      { return _mm_free(p); }
#elif defined (_MSC_VER)
  #include <malloc.h>
  inline void *aligned_malloc (size_t size, size_t align=16) { return _aligned_malloc(size,align);  }
  inline void  aligned_free   (void *p)                      { return _aligned_free(p); }
#elif defined (__CYGWIN__)
  #include <xmmintrin.h>
  inline void *aligned_malloc (size_t size, size_t align=16) { return _mm_malloc(size,align);  }
  inline void  aligned_free   (void *p)                      { return _mm_free(p); }
#elif defined(__MINGW64__)
  #include <malloc.h>
  inline void *aligned_malloc (size_t size, size_t align=16) { return __mingw_aligned_malloc(size,align);  }
  inline void  aligned_free   (void *p)                      { return __mingw_aligned_free(p); }
#elif defined(__MINGW32__)
  #include <malloc.h>
  inline void *aligned_malloc (size_t size, size_t align=16) { return __mingw_aligned_malloc(size,align);  }
  inline void  aligned_free   (void *p)                      { return __mingw_aligned_free(p); }
#elif defined(__FreeBSD__)
  #include <stdlib.h>
  inline void* aligned_malloc (size_t size, size_t align=16) { return malloc(size); }
  inline void  aligned_free   (void *p)                      { return free(p); }
#elif (defined(__MACOSX__) || defined(__APPLE__))
  #include <stdlib.h>
  inline void* aligned_malloc (size_t size, size_t align=16) { return malloc(size); }
  inline void  aligned_free   (void *p)                      { return free(p); }
#else
  #include <malloc.h>
  inline void* aligned_malloc (size_t size, size_t align=16) { return memalign(align,size); }
  inline void  aligned_free   (void *p)                      { return free(p); }
#endif

namespace smil
{
#ifdef __AVX__  
    #define SIMD_VEC_SIZE 32
#else // __AVX__  
    #define SIMD_VEC_SIZE 16
#endif // __AVX__
  
  
    #define ASSUME_ALIGNED(buf) __builtin_assume_aligned(buf, SIMD_VEC_SIZE)


    template<typename T> 
    inline T *createAlignedBuffer(size_t size) 
    {
      void* ptr;

      ptr = aligned_malloc((SIMD_VEC_SIZE*(size/SIMD_VEC_SIZE+1))*sizeof(T), SIMD_VEC_SIZE);

      return ((T*) (ptr));
    }

    template<typename T> 
    void deleteAlignedBuffer(T *ptr) {
      aligned_free( (void*)(ptr) );
    }

    template<typename T> 
    inline void Dmemcpy(T *out, const T *in, size_t size)
    {
	while (size--)
	{
	    *out++ = *in++;
	}
    }


    template<typename T> 
    void t_LineCopyFromImage2D(T *rawImagePointerIn, const size_t lineSize, size_t y, T *lineout) {

      T *ptrin = rawImagePointerIn + y*lineSize;

      memcpy(lineout,ptrin,lineSize*sizeof(T));

    }


    template<typename T> 
    void t_LineCopyToImage2D(T *linein, const size_t lineSize, size_t y, T *rawImagePointerOut) {

      T *ptrout = rawImagePointerOut + y*lineSize;

      memcpy(ptrout,linein,lineSize*sizeof(T));

    }

    template<typename T> 
    void t_LineShiftRight(const T *linein, const int lineWidth, const int nbshift, const T shiftValue, T *lineout) 
    {
      int i;

      for(i=0 ; i<nbshift ; i++)  
      {
	lineout[i] = shiftValue;
      }

      memcpy(lineout+nbshift,linein,(lineWidth-nbshift)*sizeof(T));

    }


    template<typename T> 
    void t_LineShiftLeft(const T *linein, const int lineWidth, const int nbshift, const T shiftValue, T *lineout) 
    {
      int i;

      for(i=lineWidth-nbshift ; i<lineWidth ; i++)  
      {
	lineout[i] = shiftValue;
      }

      memcpy(lineout,linein+nbshift,(lineWidth-nbshift)*sizeof(T));

    }

    inline size_t PTR_OFFSET(void *p, size_t n=SIMD_VEC_SIZE)
    {
	return ((size_t)p) & (n-1);
    }

    inline std::string displayBytes(size_t bytes)
    {
	    char tmp[128] = "";
	    const char *units[] = { "B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB" };
	    const double base = 1024;
	    int c = std::min((int)(log((double)bytes)/log(base)), (int)sizeof(units) - 1);
	    sprintf(tmp, "%1.2f %s", bytes / pow(base, c), units[c]);
	    return std::string(tmp);
    }
    

} // namespace smil

#endif // _DMEMORY_HPP

