#ifndef _D_LINE_ARITH_HPP
#define _D_LINE_ARITH_HPP


#include "DImage.hpp"
#include "DBaseVectorOperations.hpp"

template <class T>
struct copyPixel
{
    inline void _exec(T &pIn, T &pOut)
    {
	pOut = pIn;
    }
};

// #define fillLine unaryLineFunction<T, copyPixel<T> > 

// template <class T>
// typedef unaryLineFunction<T, copyPixel<T> > filLine;

/*template <class T>
struct arith
{
  typedef unaryLineFunction<T, copyPixel<T> > fillLine;
};*/
// 
template <class T>
struct fillLine : public unaryLineFunction<T, copyPixel<T> >
{
    fillLine(T *lInOut, int size, T value)
    {
	_exec(lInOut, size, value);
    }
};

// template <class T>
// struct fillLine : public unaryLineFunctionBase<T>
// {
//     inline void _exec(T *lInOut, int size, T value)
//     {
// 	for (int i=0;i<size;i++)
// 	  lInOut[i] = value;
//     }
// };


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
};


template <class T>
struct addNoSatLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lIn1[i] + lIn2[i];
    }
};

template <class T>
struct subLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < (T)(numeric_limits<T>::max() + lIn2[i]) ? numeric_limits<T>::min() : lIn1[i] - lIn2[i];
    }
};

template <class T>
struct subNoSatLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lIn1[i] - lIn2[i];
    }
};

template <class T>
struct supLine : public binaryLineFunctionBase<T> 
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for(int i=0;i<size;i++)
	  lOut[i] = lIn1[i]>lIn2[i] ? lIn1[i] : lIn2[i];
    }
};

// template <class T>
// struct supPixel
// {
//     inline void _exec(T &pIn1, T &pIn2, T &pOut)
//     {
// 	pOut = pIn1 > pIn2 ? pIn1 : pIn2;
//     }
// };
// 
// template <class T>
// struct supLine : public binaryLineFunction<T, supPixel<T> >
// {
// };


template <class T>
struct infLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < lIn2[i] ? lIn1[i] : lIn2[i];
    }
};

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
struct lowLine : public binaryLineFunctionBase<T>
{
    inline void _exec(T *lIn1, T *lIn2, int size, T *lOut)
    {
	for (int i=0;i<size;i++)
	  lOut[i] = lIn1[i] < lIn2[i] ? numeric_limits<T>::max() : 0;
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
};

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




#endif // _D_LINE_ARITH_HPP
