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


#ifndef _D_IMAGE_TRANSFORM_HPP
#define _D_IMAGE_TRANSFORM_HPP

#include "DBaseImageOperations.hpp"
#include "DLineArith.hpp"

/**
 * \ingroup Base
 * \defgroup Transform
 * @{
 */


/**
 * Vertical flip (horizontal mirror).
 * 
 * Quick implementation (needs better integration and optimization).
 */
template <class T>
RES_T vFlip(Image<T> &imIn, Image<T> &imOut)
{
    if (&imIn==&imOut)
	return vFlip(imIn);
    
    if (!areAllocated(&imIn, &imOut, NULL))
        return RES_ERR_BAD_ALLOCATION;
  
    if (!haveSameSize(&imIn, &imOut, NULL))
        return RES_ERR;
  
    typename Image<T>::sliceType *slicesIn = imIn.getSlices();
    typename Image<T>::sliceType *slicesOut = imOut.getSlices();
    typename Image<T>::sliceType linesIn;
    typename Image<T>::sliceType linesOut;
    
    UINT width = imIn.getWidth();
    UINT height = imIn.getHeight();
    UINT depth = imIn.getDepth();

    for (UINT k=0;k<depth;k++)
    {
	linesIn = slicesIn[k];
	linesOut = slicesOut[k];
	
	for (UINT j=0;j<height;j++)
	  copyLine<T>(linesIn[j], width, linesOut[height-1-j]);
    }
    
    imOut.modified();
    
    return RES_OK;
}

template <class T>
RES_T vFlip(Image<T> &imInOut)
{
    if (!imInOut.isAllocated())
        return RES_ERR_BAD_ALLOCATION;
  
    typename Image<T>::sliceType *slicesIn = imInOut.getSlices();
    typename Image<T>::sliceType linesIn;
    
    UINT width = imInOut.getWidth();
    UINT height = imInOut.getHeight();
    UINT depth = imInOut.getDepth();

    typename Image<T>::lineType tmpLine = ImDtTypes<T>::createLine(width);
      
    for (UINT k=0;k<depth;k++)
    {
	linesIn = slicesIn[k];
	
	for (UINT j=0;j<height/2;j++)
	{
	    copyLine<T>(linesIn[j], width, tmpLine);
	    copyLine<T>(linesIn[height-1-j], width, linesIn[j]);
	    copyLine<T>(tmpLine, width, linesIn[height-1-j]);
	}
    }
    
    ImDtTypes<T>::deleteLine(tmpLine);
    imInOut.modified();
    return RES_OK;
}

/**
 * Image translation.
 * 
 */
template <class T>
RES_T trans(Image<T> &imIn, int dx, int dy, int dz, Image<T> &imOut, T borderValue = numeric_limits<T>::min())
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;
    
    UINT lineLen = imIn.getWidth();
    typename ImDtTypes<T>::lineType borderBuf = ImDtTypes<T>::createLine(lineLen);
    fillLine<T>(borderBuf, lineLen, borderValue);
    
    UINT height = imIn.getHeight();
    UINT depth  = imIn.getDepth();
    
    for (UINT k=0;k<depth;k++)
    {
	typename Image<T>::sliceType lOut = imOut.getSlices()[k];
	
	int z = k+dz;
	for (UINT j=0;j<height;j++, lOut++)
	{
	    int y = j+dy;
	    
	    if (z<0 || z>=(int)depth || y<0 || y>=(int)height)
		copyLine<T>(borderBuf, lineLen, *lOut);
	    else 
		shiftLine<T>(imIn.getSlices()[z][y], dx, lineLen, *lOut, borderValue);
	}
    }
    
    ImDtTypes<T>::deleteLine(borderBuf);
    
    imOut.modified();
    
    return RES_OK;
}

template <class T>
RES_T trans(Image<T> &imIn, int dx, int dy, Image<T> &imOut, T borderValue = numeric_limits<T>::min())
{
    return trans<T>(imIn, dx, dy, 0, imOut, borderValue);
}

template <class T>
Image<T> trans(Image<T> &imIn, int dx, int dy, int dz)
{
    Image<T> imOut(imIn);
    trans<T>(imIn, dx, dy, dz, imOut);
    return imOut;
}

template <class T>
Image<T> trans(Image<T> &imIn, int dx, int dy)
{
    Image<T> imOut(imIn);
    trans<T>(imIn, dx, dy, 0, imOut);
    return imOut;
}


/**
 * 2D bilinear resize algorithm.
 * 
 * Quick implementation (needs better integration and optimization).
 */
template <class T>
RES_T resize(Image<T> &imIn, Image<T> &imOut)
{
    if (!imIn.isAllocated() || !imOut.isAllocated())
        return RES_ERR_BAD_ALLOCATION;
  
    UINT w = imIn.getWidth();
    UINT h = imIn.getHeight();
    
    UINT w2 = imOut.getWidth();
    UINT h2 = imOut.getHeight();
    
    typedef typename Image<T>::pixelType pixelType;
    typedef typename Image<T>::lineType lineType;
    
    lineType pixIn = imIn.getPixels();
    lineType pixOut = imOut.getPixels();
    
    pixelType A, B, C, D, maxVal = numeric_limits<T>::max() ;
    int x, y, index;
    
    float x_ratio = ((float)(w-1))/w2 ;
    float y_ratio = ((float)(h-1))/h2 ;
    float x_diff, y_diff;
    int offset = 0 ;
    
    for (UINT i=0;i<h2;i++) 
    {
        for (UINT j=0;j<w2;j++) 
	{
            x = (int)(x_ratio * j) ;
            y = (int)(y_ratio * i) ;
            x_diff = (x_ratio * j) - x ;
            y_diff = (y_ratio * i) - y ;
            index = y*w+x ;

            A = pixIn[index] & maxVal ;
            B = pixIn[index+1] & maxVal ;
            C = pixIn[index+w] & maxVal ;
            D = pixIn[index+w+1] & maxVal ;
            
            // Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
            pixOut[offset++] = A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) + C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff);
        }
    }
    imOut.modified();
    
    return RES_OK;
}


/** @}*/

#endif // _D_IMAGE_TRANSFORM_HPP

