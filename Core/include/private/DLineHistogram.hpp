#ifndef _D_LINE_HISTOGRAM_HPP
#define _D_LINE_HISTOGRAM_HPP


#include "DImage.hpp"

//! \ingroup Core
//! \defgroup Histogram
//! @{

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

template <class T>
struct stretchHistLine : public unaryLineFunctionBase<T>
{
    T inOrig, outOrig;
    double coeff;
    
    inline void _exec(T* lIn, int size, T* lOut)
    {
	double newVal;
	
	for(int i=0;i<size;i++)
	{
	    newVal = outOrig + (lIn[i]-inOrig)*coeff;
	    if (newVal > numeric_limits<T>::max())
		newVal = numeric_limits<T>::max();
	    else if (newVal < numeric_limits<T>::min())
		newVal = numeric_limits<T>::min();
	    lOut[i] = T(newVal);
	    
	}
    }
};


//! @}

#endif // _D_LINE_HISTOGRAM_HPP
