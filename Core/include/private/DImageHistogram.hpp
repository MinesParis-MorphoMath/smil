#ifndef _D_IMAGE_HISTOGRAM_HPP
#define _D_IMAGE_HISTOGRAM_HPP

#include "DLineHistogram.hpp"

//! \ingroup Core
//! \ingroup Histogram
//! @{

//! Image threshold
template <class T>
inline RES_T thresh(Image<T> &imIn, T minVal, T maxVal, T trueVal, T falseVal, Image<T> &imOut)
{
    unaryImageFunction<T, threshLine<T> > iFunc;
    
    iFunc.lineFunction.minVal = minVal;
    iFunc.lineFunction.maxVal = maxVal;
    iFunc.lineFunction.trueVal = trueVal;
    iFunc.lineFunction.falseVal = falseVal;
    
    return iFunc(imIn, imOut);
}

template <class T>
inline RES_T thresh(Image<T> &imIn, T maxVal, Image<T> &imOut)
{
    unaryImageFunction<T, threshLine<T> > iFunc;
    
    iFunc.lineFunction.minVal = numeric_limits<T>::min();
    iFunc.lineFunction.maxVal = maxVal;
    iFunc.lineFunction.trueVal = numeric_limits<T>::max();
    iFunc.lineFunction.falseVal = numeric_limits<T>::min();
    
    return iFunc(imIn, imOut);
}


#endif // _D_IMAGE_HISTOGRAM_HPP

