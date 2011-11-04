#ifndef _D_LINE_HISTOGRAM_HPP
#define _D_LINE_HISTOGRAM_HPP


#include "DImage.hpp"

template <class T>
struct threshLine : public unaryLineFunctionBase<T>
{
    T minVal, maxVal, trueVal, falseVal;
    
    inline void _exec(T* lIn, int size, T* lOut)
    {
	for(int i=0;i<size;i++)
	    lOut[i] = lIn[i] >= minVal && lIn[i] <= maxVal  ? trueVal : falseVal;
    }
};



#endif // _D_LINE_HISTOGRAM_HPP
