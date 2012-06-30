/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _D_IMAGE_HISTOGRAM_HPP
#define _D_IMAGE_HISTOGRAM_HPP

#include "DLineHistogram.hpp"
#include "DImageArith.hpp"

//! \ingroup Core
//! \ingroup Histogram
//! @{

template <class T>
vector<UINT> histo(Image<T> &imIn)
{
    T dx = ImDtTypes<T>::min();
    vector<UINT> h;
    for (int i=0;i<ImDtTypes<T>::max()-dx;i++)
	h.push_back(0);
    
    typename Image<T>::lineType pixels = imIn.getPixels();
    for (int i=0;i<imIn.getPixelCount();i++)
	h[pixels[i]-dx] += 1;
    
    return h;
}

//! Image threshold
template <class T>
RES_T thresh(Image<T> &imIn, T minVal, T maxVal, T trueVal, T falseVal, Image<T> &imOut)
{
    unaryImageFunction<T, threshLine<T> > iFunc;
    
    iFunc.lineFunction.minVal = minVal;
    iFunc.lineFunction.maxVal = maxVal;
    iFunc.lineFunction.trueVal = trueVal;
    iFunc.lineFunction.falseVal = falseVal;
    
    return iFunc(imIn, imOut);
}

template <class T>
RES_T thresh(Image<T> &imIn, T minVal, T maxVal, Image<T> &imOut)
{
    unaryImageFunction<T, threshLine<T> > iFunc;
    
    iFunc.lineFunction.minVal = minVal;
    iFunc.lineFunction.maxVal = maxVal;
    iFunc.lineFunction.trueVal = numeric_limits<T>::max();
    iFunc.lineFunction.falseVal = numeric_limits<T>::min();
    
    return iFunc(imIn, imOut);
}

template <class T>
RES_T thresh(Image<T> &imIn, T minVal, Image<T> &imOut)
{
    unaryImageFunction<T, threshLine<T> > iFunc;
    
    iFunc.lineFunction.minVal = minVal;
    iFunc.lineFunction.maxVal = numeric_limits<T>::max();
    iFunc.lineFunction.trueVal = numeric_limits<T>::max();
    iFunc.lineFunction.falseVal = numeric_limits<T>::min();
    
    return iFunc(imIn, imOut);
}

/**
 * 
 */
template <class T>
RES_T stretchHist(Image<T> &imIn, T inMinVal, T inMaxVal, Image<T> &imOut, T outMinVal=numeric_limits<T>::min(), T outMaxVal=numeric_limits<T>::max())
{
    unaryImageFunction<T, stretchHistLine<T> > iFunc;
    iFunc.lineFunction.coeff = double (outMaxVal-outMinVal) / double (inMaxVal-inMinVal);
    iFunc.lineFunction.inOrig = inMinVal;
    iFunc.lineFunction.outOrig = outMinVal;
    
    return iFunc(imIn, imOut);
}

/**
 * 
 */
template <class T>
RES_T stretchHist(Image<T> &imIn, Image<T> &imOut, T outMinVal=numeric_limits<T>::min(), T outMaxVal=numeric_limits<T>::max())
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
RES_T enhanceContrast(Image<T> &imIn, Image<T> &imOut, double sat=0.5)
{
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;
    
    vector<UINT> h = histo(imIn);
    double imVol = imIn.getWidth() * imIn.getHeight() * imIn.getDepth();
    double satVol = imVol * sat / 100.;
    double v = 0;
    T dx = ImDtTypes<T>::min();
    T minV, maxV;
    rangeVal(imIn, &minV, &maxV);
    
    for (UINT i=ImDtTypes<T>::max()-1-dx; i>0; i--)
    {
	v += h[i];
	if (v>satVol)
	    break;
	maxV = i + dx;
    }
    
    stretchHist(imIn, minV, maxV, imOut);
    imOut.modified();
    
    return RES_OK;
}



#endif // _D_IMAGE_HISTOGRAM_HPP

