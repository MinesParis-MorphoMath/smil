#ifndef _D_LINE_ARITH_HPP
#define _D_LINE_ARITH_HPP


#include "DImage.hpp"

#ifdef __SSE__
#include <emmintrin.h>
#endif // __SSE__


/**
 * \ingroup Core
 * \defgroup Arith
 * @{
 */


template <class T>
struct fillLine : public unaryLineFunctionBase<T>
{
    inline void _exec(T *lInOut, int size, T value)
    {
	for (int i=0;i<size;i++)
	  lInOut[i] = value;
    }
};

template <class T>
struct invLine : public unaryLineFunctionBase<T>
{
    inline void _exec(T *lineIn, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = ~lineIn[i];
    }
};

template <class T>
struct addLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T* lIn1, T* lIn2, int size, T* lOut)
    {
	for(int i=0;i<size;i++)
	    lOut[i] = lIn1[i] > (T)(numeric_limits<T>::max()- lIn2[i]) ? numeric_limits<T>::max() : lIn1[i] + lIn2[i];
    }
    inline void _exec_aligned(T *lIn1, T *lIn2, int size, T *lOut) { _exec(lIn1, lIn2, size, lOut); }
};

#ifdef __SSE__

template <>
inline void addLine<UINT8>::_exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
{
     __m128i r0,r1;
    __m128i *l1 = (__m128i*) lIn1;
    __m128i *l2 = (__m128i*) lIn2;
    __m128i *l3 = (__m128i*) lOut;
    for(int i=0 ; i<size ; i+=16, l1++, l2++, l3++)
    {
	r0 = _mm_load_si128(l1);
	r1 = _mm_load_si128(l2);
	_mm_adds_epu8(r0, r1);
	_mm_store_si128(l3, r1);
    }
}

#endif // __SSE__


template <class T>
struct addNoSatLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] + lIn2[i];
    }
    inline void _exec_aligned(T *lIn1, T *lIn2, int size, T *lOut) { _exec(lIn1, lIn2, size, lOut); }
};

#ifdef __SSE__

template <>
inline void addNoSatLine<UINT8>::_exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
{
     __m128i r0,r1;
    __m128i *l1 = (__m128i*) lIn1;
    __m128i *l2 = (__m128i*) lIn2;
    __m128i *l3 = (__m128i*) lOut;
    for(int i=0 ; i<size ; i+=16, l1++, l2++, l3++)
    {
	r0 = _mm_load_si128(l1);
	r1 = _mm_load_si128(l2);
	_mm_add_epi8(r0, r1);
	_mm_store_si128(l3, r1);
    }
}

#endif // __SSE__

template <class T>
struct subLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < (T)(numeric_limits<T>::max() + lIn2[i]) ? numeric_limits<T>::min() : lIn1[i] - lIn2[i];
    }
    inline void _exec_aligned(T *lIn1, T *lIn2, int size, T *lOut) { _exec(lIn1, lIn2, size, lOut); }
};

#ifdef __SSE__

template <>
inline void subLine<UINT8>::_exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
{
     __m128i r0,r1;
    __m128i *l1 = (__m128i*) lIn1;
    __m128i *l2 = (__m128i*) lIn2;
    __m128i *l3 = (__m128i*) lOut;
    for(int i=0 ; i<size ; i+=16, l1++, l2++, l3++)
    {
	r0 = _mm_load_si128(l1);
	r1 = _mm_load_si128(l2);
	_mm_subs_epu8(r0, r1);
	_mm_store_si128(l3, r1);
    }
}

#endif // __SSE__

template <class T>
struct subNoSatLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] - lIn2[i];
    }
    inline void _exec_aligned(T *lIn1, T *lIn2, int size, T *lOut) { _exec(lIn1, lIn2, size, lOut); }
};

#ifdef __SSE__

template <>
inline void subNoSatLine<UINT8>::_exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
{
     __m128i r0,r1;
    __m128i *l1 = (__m128i*) lIn1;
    __m128i *l2 = (__m128i*) lIn2;
    __m128i *l3 = (__m128i*) lOut;
    for(int i=0 ; i<size ; i+=16, l1++, l2++, l3++)
    {
	r0 = _mm_load_si128(l1);
	r1 = _mm_load_si128(l2);
	_mm_sub_epi8(r0, r1);
	_mm_store_si128(l3, r1);
    }
}

#endif // __SSE__

template <class T>
struct supLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] > lIn2[i] ? lIn1[i] : lIn2[i];
    }
    inline void _exec_aligned(T *lIn1, T *lIn2, int size, T *lOut) { _exec(lIn1, lIn2, size, lOut); }
};

#ifdef __SSE__

template <>
inline void supLine<UINT8>::_exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
{
     __m128i r0,r1;
    __m128i *l1 = (__m128i*) lIn1;
    __m128i *l2 = (__m128i*) lIn2;
    __m128i *l3 = (__m128i*) lOut;
    for(int i=0 ; i<size ; i+=16, l1++, l2++, l3++)
    {
	r0 = _mm_load_si128(l1);
	r1 = _mm_load_si128(l2);
	_mm_max_epu8(r0, r1);
	_mm_store_si128(l3, r1);
    }
}

#endif // __SSE__

template <class T>
struct infLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < lIn2[i] ? lIn1[i] : lIn2[i];
    }
    inline void _exec_aligned(T *lIn1, T *lIn2, int size, T *lOut) { _exec(lIn1, lIn2, size, lOut); }
};

#ifdef __SSE__

template <>
inline void infLine<UINT8>::_exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
{
     __m128i r0,r1;
    __m128i *l1 = (__m128i*) lIn1;
    __m128i *l2 = (__m128i*) lIn2;
    __m128i *l3 = (__m128i*) lOut;
    for(int i=0 ; i<size ; i+=16, l1++, l2++, l3++)
    {
	r0 = _mm_load_si128((__m128i*) lIn1);
	r1 = _mm_load_si128((__m128i*) lIn2);
	_mm_min_epu8(r0, r1);
	_mm_store_si128((__m128i*) lOut, r1);
    }
}

#endif // __SSE__


template <class T>
struct grtLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] > lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct grtOrEquLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] >= lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct lowLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct lowOrEquLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] <= lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct equLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] == lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct difLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] != lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};


template <class T>
struct mulLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	    lOut[i] = double(lIn1[i]) * double(lIn2[i]) > double(numeric_limits<T>::max()) ? numeric_limits<T>::max() : lIn1[i] * lIn2[i];
    }
    inline void _exec_aligned(T *lIn1, T *lIn2, int size, T *lOut) { _exec(lIn1, lIn2, size, lOut); }
};

#ifdef __SSE__

template <>
inline void mulLine<UINT8>::_exec_aligned(UINT8 *lIn1, UINT8 *lIn2, int size, UINT8 *lOut)
{
     __m128i r0,r1;
    __m128i *l1 = (__m128i*) lIn1;
    __m128i *l2 = (__m128i*) lIn2;
    __m128i *l3 = (__m128i*) lOut;
    for(int i=0 ; i<size ; i+=16, l1++, l2++, l3++)
    {
	r0 = _mm_load_si128(l1);
	r1 = _mm_load_si128(l2);
	_mm_mullo_epi16(r0, r1);
	_mm_store_si128(l3, r1);
    }
}

#endif // __SSE__

template <class T>
struct mulNoSatLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	    lOut[i] = (T)(lIn1[i] * lIn2[i]);
    }
};

template <class T>
struct divLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	{
	    lOut[i] = lIn2[i]==0 ? numeric_limits<T>::max() : lIn1[i] / lIn2[i];
	}
    }
};


template <class T>
struct testLine : public tertiaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, T *lIn3, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	{
	    lOut[i] = lIn1[i] ? lIn2[i] : lIn3[i];
	}
    }
};


/** @}*/

#endif // _D_LINE_ARITH_HPP
