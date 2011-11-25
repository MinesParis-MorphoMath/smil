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


#ifndef _DMEMORY_HPP
#define _DMEMORY_HPP


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


#define SIMD_VEC_SIZE 16

template<typename T> 
T *createAlignedBuffer(int size) {
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

