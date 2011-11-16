#ifndef _BASE_LINE_OPERATIONS_HPP
#define _BASE_LINE_OPERATIONS_HPP


#include "DImage.hpp"

struct stat;

template <class T> class Image;



// Base abstract struct of line unary function
template <class T>
struct _SMIL unaryLineFunctionBase
{
    virtual void _exec(T *lineIn, int size, T *lineOut) {}
    virtual void _exec(T *lInOut, int size, T value) {}
    inline void operator()(T *lineIn, int size, T *lineOut) { _exec(lineIn, size, lineOut); }
    inline void operator()(T *lineIn, int size, T value) { _exec(lineIn, size, value); }
};


// Base abstract struct of line binary function
template <class T>
struct _SMIL binaryLineFunctionBase
{
    virtual void _exec(T *lineIn1, T *lineIn2, int size, T *lineOut) {}
    virtual void _exec_aligned(T *lineIn1, T *lineIn2, int size, T *lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
    inline void operator()(T *lineIn1, T *lineIn2, int size, T *lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
};

// Base abstract struct of line binary function
template <class T>
struct _SMIL tertiaryLineFunctionBase
{
    virtual void _exec(T *lineIn1, T *lineIn2, T *lineIn3, int size, T *lineOut) {}
    virtual void _exec_aligned(T *lineIn1, T *lineIn2, T *lineIn3, int size, T *lineOut) { _exec(lineIn1, lineIn2, lineIn3, size, lineOut); }
    inline void operator()(T *lineIn1, T *lineIn2, T *lineIn3, int size, T *lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
};



#endif
