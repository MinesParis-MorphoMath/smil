#ifndef _BASE_LINE_OPERATIONS_HPP
#define _BASE_LINE_OPERATIONS_HPP


#include "DImage.hpp"

struct stat;

template <class T> class Image;



// Base abstract struct of line unary function
template <class T>
struct _SMIL unaryLineFunctionBase
{
    static void _exec(T *lineIn, int size, T *lineOut) {};
    inline void operator()(T *lineIn, int size, T *lineOut) { _exec(lineIn, size, lineOut); }
};


// Base abstract struct of line binary function
template <class T>
struct _SMIL binaryLineFunctionBase
{
    static void _exec(T *lineIn1, T *lineIn2, int size, T *lineOut);
    inline void operator()(T *lineIn1, T *lineIn2, int size, T *lineOut) { _exec(lineIn1, lineIn2, size, lineOut); }
};



#endif
