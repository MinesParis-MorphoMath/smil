#ifndef _BASE_LINE_OPERATIONS_HPP
#define _BASE_LINE_OPERATIONS_HPP


#include "DImage.hpp"
#include "DBasePixelOperations.hpp"

struct stat;

template <class T> class Image;



// Base abstract struct of line unary function
template <class T>
struct _SMIL unaryLineFunctionBase
{
    virtual void _exec(T *lineIn, int size, T *lineOut) = 0;
    virtual inline void operator()(T *lineIn, int size, T *lineOut) { _exec(lineIn, size, lineOut); }
};


// Base abstract struct of line binary function
template <class T>
struct _SMIL binaryLineFunctionBase
{
    virtual void _exec(T *lineIn1, T *lineIn2, int size, T *lineOut) = 0;
    virtual inline void operator()(T *lineIn1, T *lineIn2, int size, T *lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
};

class lineFunctionBase
{
};

template <class T, class unaryPixelFunction_T>
struct unaryLineFunction
{
    static unaryPixelFunction_T pixelFunction;
    
    static void _exec(T *lineIn, int size, T *lineOut)
    {
	for(int i=0;i<size;i++)
	  pixelFunction._exec(lineIn[i], lineOut[i]);
    }
    static void _exec(T *lineInOut, int size, T value)
    {
	for(int i=0;i<size;i++)
	  pixelFunction._exec(value, lineInOut[i]);
    }
    
    inline void operator()(T *lineIn, int size, T *lineOut) { _exec(lineIn, size, lineOut); }
    inline void operator()(T *lineInOut, int size, T value) { _exec(lineInOut, size, value); }
    unaryLineFunction() {}
    unaryLineFunction(T *lineIn, int size, T *lineOut) { _exec(lineIn, size, lineOut); }
    unaryLineFunction(T *lineInOut, int size, T value) { _exec(lineInOut, size, value); }
};


template <class T, class binaryPixelFunction_T>
struct binaryLineFunction
{
    static binaryPixelFunction_T pixelFunction;
    static void _exec(T *lineIn1, T *lineIn2, int size, T *lineOut)
    {
	for(int i=0;i<size;i++)
	  pixelFunction._exec(lineIn1[i], lineIn2[i], lineOut[i]);
    }
    static void _exec(T *lineIn, T value, int size, T *lineOut)
    {
	for(int i=0;i<size;i++)
	  pixelFunction._exec(lineIn[i], value, lineOut[i]);
    }
    inline void operator()(T *lineIn1, T *lineIn2, int size, T *lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
    inline void operator()(T *lineIn, T value, int size, T *lineOut) { _exec(lineIn, value, size, lineOut); }
};


#endif
