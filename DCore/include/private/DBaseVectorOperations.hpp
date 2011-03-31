#ifndef _BASE_VECTOR_OPERATIONS_HPP
#define _BASE_VECTOR_OPERATIONS_HPP


#include "DImage.hpp"


template <class T> class Image;


template <class T, class vectFunc >
struct binaryVectFunction : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lineIn1, T *lineIn2, int size, T *lineOut)
    {
	cout << "Not implemented" << endl;
    }
};

#ifdef __SSE__

template <class vectFunc >
struct binaryVectFunction<UINT8, vectFunc > : public binaryLineFunctionBase<UINT8>
{
     vectFunc f;
    __m128i r0,r1;
    
    inline void _exec(UINT8 *lineIn1, UINT8 *lineIn2, int size, UINT8 *lineOut)
    {
	for(int i=0 ; i<size+16 ; i+=16)  
	{
	  r0 = _mm_load_si128((__m128i *) lineIn1);
	  r1 = _mm_load_si128((__m128i *) lineIn2);
	  _mm_add_epi8(r0, r1);
	  f._exec_UINT8(r0, r1);
	  _mm_store_si128((__m128i *) lineOut,r1);

	  lineIn1 += 16;
	  lineIn2 += 16;
	  lineOut += 16;
	}
    }
};

template <class vectFunc >
struct binaryVectFunction<UINT16, vectFunc > : public binaryLineFunctionBase<UINT16>
{
     vectFunc f;
    __m128i r0,r1;
    
    inline void _exec(UINT16 *lineIn1, UINT16 *lineIn2, int size, UINT16 *lineOut)
    {
	for(int i=0 ; i<size+16 ; i+=8)  
	{
	  r0 = _mm_load_si128((__m128i *) lineIn1);
	  r1 = _mm_load_si128((__m128i *) lineIn2);
	  f._exec_UINT16(r0, r1);
	  _mm_store_si128((__m128i *) lineOut,r1);

	  lineIn1 += 8;
	  lineIn2 += 8;
	  lineOut += 8;
	}
    }
};






struct addVect
{
     inline void _exec_UINT8(__m128i &r0, __m128i &r1) { _mm_adds_epi8(r0, r1); }
     inline void _exec_UINT16(__m128i &r0, __m128i &r1) { _mm_adds_epi16(r0, r1); }
};


#endif // __SSE__


#endif // _BASE_VECTOR_OPERATIONS_HPP
