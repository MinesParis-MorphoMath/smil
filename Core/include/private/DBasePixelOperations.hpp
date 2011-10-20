#ifndef _BASE_PIXEL_OPERATIONS_HPP
#define _BASE_PIXEL_OPERATIONS_HPP


#ifdef __SSE__
#include <mmintrin.h>
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif



template <class T>
struct basePixelFunction
{
};

template <class T>
struct unaryPixelFunction
{
    // use of a real function name (_exec) instead of usual operator() to aid auto-vectorization
    virtual void _exec(T &pIn, T &pOut) = 0;
};

template <class T>
struct binaryPixelFunction //: public basePixelFunction<T>
{
    // use of a real function name (_exec) instead of usual operator() to aid auto-vectorization
    virtual void _exec(T &pIn1, T &pIn2, T &pOut) = 0;
    inline void operator()(T &pIn1, T &pIn2, T &pOut) { return _exec(pIn1, pIn2, pOut); }
};








template <class T>
struct greaterPixel : public binaryPixelFunction<T>
{
    inline void _exec(T &pIn1, T &pIn2, T &pOut)
    {
	pOut = pIn1 > pIn2 ? numeric_limits<T>::max() : 0;
    }
};

template <class T>
struct subLoopPixel : public binaryPixelFunction<T>
{
    inline void _exec(T &pIn1, T &pIn2, T &pOut)
    {
	pOut = pIn1 - pIn2;
    }
};



// template <class T>
// struct supPixel : public binaryPixelFunction<T>
// {
//     inline void _exec(T &pIn1, T &pIn2, T &pOut)
//     {
// 	pOut = pIn1 > pIn2 ? pIn1 : pIn2;
//     }
// };



#endif // _BASE_PIXEL_OPERATIONS_HPP
