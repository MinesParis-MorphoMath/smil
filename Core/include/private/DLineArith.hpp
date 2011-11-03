#ifndef _D_LINE_ARITH_HPP
#define _D_LINE_ARITH_HPP


#include "DImage.hpp"

struct stat;

template <class T>
struct fillLine : public unaryLineFunctionBase<T>
{
    static void _exec(T *lInOut, int size, T value)
    {
	for (int i=0;i<size;i++)
	  lInOut[i] = value;
    }
};

template <class T>
struct invLine : public unaryLineFunctionBase<T>
{
    static void _exec(T *lineIn, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = ~lineIn[i];
    }
};

template <class T>
struct addLine : public binaryLineFunctionBase<T>
{
    static void _exec(T* lIn1, T* lIn2, int size, T* lOut)
    {
	for(int i=0;i<size;i++)
	    lOut[i] = lIn1[i] > (T)(numeric_limits<T>::max()- lIn2[i]) ? numeric_limits<T>::max() : lIn1[i] + lIn2[i];
    }
};


template <class T>
struct addNoSatLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] + lIn2[i];
    }
};

template <class T>
struct subLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < (T)(numeric_limits<T>::max() + lIn2[i]) ? numeric_limits<T>::min() : lIn1[i] - lIn2[i];
    }
};

template <class T>
struct subNoSatLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] - lIn2[i];
    }
};

template <class T>
struct supLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] > lIn2[i] ? lIn1[i] : lIn2[i];
    }
};


template <class T>
struct infLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < lIn2[i] ? lIn1[i] : lIn2[i];
    }
};

template <class T>
struct grtLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] > lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct lowLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct equLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] == lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct difLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] != lIn2[i] ? numeric_limits<T>::max() : 0;
    }
};


template <class T>
struct mulLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	    lOut[i] = double(lIn1[i]) * double(lIn2[i]) > double(numeric_limits<T>::max()) ? numeric_limits<T>::max() : lIn1[i] * lIn2[i];
    }
};

template <class T>
struct mulNoSatLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	    lOut[i] = (T)(lIn1[i] * lIn2[i]);
    }
};

template <class T>
struct divLine : public binaryLineFunctionBase<T>
{
    static void _exec(T *lIn1, T *lIn2, int size, T *lOut)
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
    static void _exec(T *lIn1, T *lIn2, T *lIn3, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	{
	    lOut[i] = lIn1[i] ? lIn2[i] : lIn3[i];
	}
    }
};




#endif // _D_LINE_ARITH_HPP
