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


#ifndef _DMEMORY_HPP
#define _DMEMORY_HPP


#if defined(__MINGW32__)

    #if __GNUC__ < 4
	#include <malloc.h>
	#define MALLOC(ptr,size, align)  ptr = __mingw_aligned_malloc(size,align)
	#define FREE(p)                  __mingw_aligned_free(p)
    #else // __GNUC__ < 4
	#include <mm_malloc.h>
	#define MALLOC(ptr,size,align)   ptr = _mm_malloc(size,align)
	#define FREE(p)                  _mm_free(p)
    #endif // __GNUC__ < 4

#elif defined(_MSC_VER)

    #include <malloc.h>
    #define MALLOC(ptr,size,align)   ptr = _aligned_malloc(size,align)
    #define FREE(p)                  _aligned_free(p)

#else

//     #include <mm_malloc.h>
//     #define MALLOC(ptr,size,align)   ptr = _mm_malloc(size,align)
//     #define FREE(p)                  _mm_free(p)
    #include <malloc.h>
    #define MALLOC(ptr,size,align)   ptr = malloc(size)
    #define FREE(p)                  free(p)

#endif


#define SIMD_VEC_SIZE 16

#define ASSUME_ALIGNED(buf) __builtin_assume_aligned(buf, SIMD_VEC_SIZE)


template<typename T> 
inline T *createAlignedBuffer(int size) {
  void* ptr;

  MALLOC(ptr,(size+32)*sizeof(T),SIMD_VEC_SIZE);
//   posix_memalign (&ptr, 16, (size+32)*sizeof(T));

  return ((T*) (ptr));
//   return new T[size];
}

template<typename T> 
void deleteAlignedBuffer(T *ptr) {
  FREE( (void*)(ptr) );
}

template<typename T> 
inline void Dmemcpy(T *out, const T *in, unsigned int size)
{
    while (size--)
    {
        *out++ = *in++;
    }
}


template<typename T> 
void t_LineCopyFromImage2D(T *rawImagePointerIn, const int lineSize, int y, T *lineout) {

  T *ptrin = rawImagePointerIn + y*lineSize;

  memcpy(lineout,ptrin,lineSize*sizeof(T));

}


template<typename T> 
void t_LineCopyToImage2D(T *linein, const int lineSize, int y, T *rawImagePointerOut) {

  T *ptrout = rawImagePointerOut + y*lineSize;

  memcpy(ptrout,linein,lineSize*sizeof(T));

}

template<typename T> 
void t_LineShiftRight1D(const T *linein, const int lineWidth, const int nbshift, const T shiftValue, T *lineout) {
  int i;

  for(i=0 ; i<nbshift ; i++)  {
    lineout[i] = shiftValue;
  }

  memcpy(lineout+nbshift,linein,(lineWidth-nbshift)*sizeof(T));

}


template<typename T> 
void t_LineShiftLeft1D(const T *linein, const int lineWidth, const int nbshift, const T shiftValue, T *lineout) {
  int i;

  for(i=lineWidth-nbshift ; i<lineWidth ; i++)  {
    lineout[i] = shiftValue;
  }

  memcpy(lineout,linein+nbshift,(lineWidth-nbshift)*sizeof(T));

}

inline unsigned long PTR_OFFSET(void *p, unsigned long n=SIMD_VEC_SIZE)
{
    return ((unsigned long)p) & (n-1);
}


#endif // _DMEMORY_HPP

