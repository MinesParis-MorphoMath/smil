/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


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
inline RES_T thresh(Image<T> &imIn, T minVal, T maxVal, Image<T> &imOut)
{
    unaryImageFunction<T, threshLine<T> > iFunc;
    
    iFunc.lineFunction.minVal = minVal;
    iFunc.lineFunction.maxVal = maxVal;
    iFunc.lineFunction.trueVal = numeric_limits<T>::max();
    iFunc.lineFunction.falseVal = numeric_limits<T>::min();
    
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

template <class T>
inline RES_T stretchHist(Image<T> &imIn, Image<T> &imOut, T outMinVal=numeric_limits<T>::min(), T outMaxVal=numeric_limits<T>::max())
{
    unaryImageFunction<T, stretchHistLine<T> > iFunc;
    T rmin, rmax;
    rangeVal(imIn, &rmin, &rmax);
    iFunc.lineFunction.coeff = double (outMaxVal-outMinVal) / double (rmax-rmin);
    iFunc.lineFunction.inOrig = rmin;
    iFunc.lineFunction.outOrig = outMinVal;
    
    return iFunc(imIn, imOut);
}

template <class T>
inline RES_T stretchHist(Image<T> &imIn, T inMinVal, T inMaxVal, Image<T> &imOut, T outMinVal=numeric_limits<T>::min(), T outMaxVal=numeric_limits<T>::max())
{
    unaryImageFunction<T, stretchHistLine<T> > iFunc;
    iFunc.lineFunction.coeff = double (outMaxVal-outMinVal) / double (inMaxVal-inMinVal);
    iFunc.lineFunction.inOrig = inMinVal;
    iFunc.lineFunction.outOrig = outMinVal;
    
    return iFunc(imIn, imOut);
}



#endif // _D_IMAGE_HISTOGRAM_HPP

