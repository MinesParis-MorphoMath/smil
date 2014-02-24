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

namespace smil
{
  
    /**
    * \ingroup Base
    * \defgroup Transform
    * @{
    */

    /**
    * Crop image
    * 
    * Crop an image into an output image
    * \param imIn input image
    * \param "startX startY [startZ]" start position of the zone in the input image
    * \param "sizeX sizeY [sizeZ]" size of the zone in the input image
    * \param imOut output image
    * 
    * \demo{copy_crop.py}
    */
    template <class T>
    RES_T crop(const Image<T> &imIn, size_t startX, size_t startY, size_t startZ, size_t sizeX, size_t sizeY, size_t sizeZ, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn);

	size_t inW = imIn.getWidth();
	size_t inH = imIn.getHeight();
	size_t inD = imIn.getDepth();
	
	size_t realSx = min(sizeX, inW-startX);
	size_t realSy = min(sizeY, inH-startY);
	size_t realSz = min(sizeZ, inD-startZ);
	
	imOut.setSize(realSx, realSy, realSz);
	return copy(imIn, startX, startY, startZ, realSx, realSy, realSz, imOut, 0, 0, 0);
    }

    template <class T>
    RES_T crop(Image<T> &imInOut, size_t startX, size_t startY, size_t startZ, size_t sizeX, size_t sizeY, size_t sizeZ)
    {
	Image<T> tmpIm(imInOut, true); // clone
	return crop(tmpIm, startX, startY, startZ, sizeX, sizeY, sizeZ, imInOut);
    }

    // 2D overload
    template <class T>
    RES_T crop(const Image<T> &imIn, size_t startX, size_t startY, size_t sizeX, size_t sizeY, Image<T> &imOut)
    {
	return crop(imIn, startX, startY, 0, sizeX, sizeY, 1, imOut);
    }

    template <class T>
    RES_T crop(Image<T> &imInOut, size_t startX, size_t startY, size_t sizeX, size_t sizeY)
    {
	return crop(imInOut, startX, startY, 0, sizeX, sizeY, 1);
    }


    /**
    * Add a border of size bSize around the original image
    * 
    */
    template <class T>
    RES_T addBorder(const Image<T> &imIn, size_t bSize, Image<T> &imOut)
    {
	Image<T> tmpIm;
	if (imIn->getDimension()==3)
	{
    // 	tmpIm.setSize(imIn.getWidth()
	}
    //     (imInOut, true); // clone
    //     return crop(tmpIm, startX, startY, startZ, sizeX, sizeY, sizeZ, imInOut);
	return RES_OK;
    }



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
	
	size_t width = imIn.getWidth();
	size_t height = imIn.getHeight();
	size_t depth = imIn.getDepth();

	for (size_t k=0;k<depth;k++)
	{
	    linesIn = slicesIn[k];
	    linesOut = slicesOut[k];
	    
	    for (size_t j=0;j<height;j++)
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
	
	size_t width = imInOut.getWidth();
	size_t height = imInOut.getHeight();
	size_t depth = imInOut.getDepth();

	typename Image<T>::lineType tmpLine = ImDtTypes<T>::createLine(width);
	  
	for (size_t k=0;k<depth;k++)
	{
	    linesIn = slicesIn[k];
	    
	    for (size_t j=0;j<height/2;j++)
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
    RES_T trans(const Image<T> &imIn, int dx, int dy, int dz, Image<T> &imOut, T borderValue = ImDtTypes<T>::min())
    {
	if (!imIn.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;
	
	size_t lineLen = imIn.getWidth();
	typename ImDtTypes<T>::lineType borderBuf = ImDtTypes<T>::createLine(lineLen);
	fillLine<T>(borderBuf, lineLen, borderValue);
	
	size_t height = imIn.getHeight();
	size_t depth  = imIn.getDepth();
	
	for (size_t k=0;k<depth;k++)
	{
	    typename Image<T>::sliceType lOut = imOut.getSlices()[k];
	    
	    int z = k-dz;
	    for (size_t j=0;j<height;j++, lOut++)
	    {
		int y = j-dy;
		
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
    RES_T trans(const Image<T> &imIn, int dx, int dy, Image<T> &imOut, T borderValue = ImDtTypes<T>::min())
    {
	return trans<T>(imIn, dx, dy, 0, imOut, borderValue);
    }

    template <class T>
    ResImage<T> trans(const Image<T> &imIn, int dx, int dy, int dz)
    {
	ResImage<T> imOut(imIn);
	trans<T>(imIn, dx, dy, dz, imOut);
	return imOut;
    }

    template <class T>
    ResImage<T> trans(const Image<T> &imIn, int dx, int dy)
    {
	ResImage<T> imOut(imIn);
	trans<T>(imIn, dx, dy, 0, imOut);
	return imOut;
    }


    /**
    * 2D bilinear resize algorithm.
    * 
    * Resize imIn to sx,sy -> imOut.
    * 
    * Quick implementation (needs better integration and optimization).
    */
    template <class T>
    RES_T resize(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut)
    {
	if (&imIn==&imOut)
	{
	    Image<T> tmpIm(imIn, true); // clone
	    return resize(tmpIm, sx, sy, imIn);
	}
	
	ImageFreezer freeze(imOut);
	
	imOut.setSize(sx, sy);
	
	if (!imIn.isAllocated() || !imOut.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;
      
	size_t w = imIn.getWidth();
	size_t h = imIn.getHeight();
	
	typedef typename Image<T>::pixelType pixelType;
	typedef typename Image<T>::lineType lineType;
	
	lineType pixIn = imIn.getPixels();
	lineType pixOut = imOut.getPixels();
	
	size_t A, B, C, D, maxVal = numeric_limits<T>::max() ;
	size_t x, y, index;
	
	double x_ratio = ((double)(w-1))/sx;
	double y_ratio = ((double)(h-1))/sy;
	double x_diff, y_diff;
	int offset = 0 ;
	
	for (size_t i=0;i<sy;i++) 
	{
	    for (size_t j=0;j<sx;j++) 
	    {
		x = (int)(x_ratio * j) ;
		y = (int)(y_ratio * i) ;
		x_diff = (x_ratio * j) - x ;
		y_diff = (y_ratio * i) - y ;
		index = y*w+x ;

		A = size_t(pixIn[index]) & maxVal ;
		B = size_t(pixIn[index+1]) & maxVal ;
		C = size_t(pixIn[index+w]) & maxVal ;
		D = size_t(pixIn[index+w+1]) & maxVal ;
		
		// Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
		pixOut[offset++] = A*(1.-x_diff)*(1.-y_diff) +  B*(x_diff)*(1.-y_diff) + C*(y_diff)*(1.-x_diff)   +  D*(x_diff*y_diff);
	    }
	}
	
	return RES_OK;
    }
    
    

    /**
    * Resize imIn with the dimensions of imOut and put the result in imOut.
    */
    template <class T>
    RES_T resize(Image<T> &imIn, Image<T> &imOut)
    {
	return resize(imIn, imOut.getWidth(), imOut.getHeight(), imOut);
    }
    
    
    /**
    * Scale image
    * If imIn has the size (W,H), the size of imOut will be (W*cx, H*cy).
    */
    template <class T>
    RES_T scale(Image<T> &imIn, double cx, double cy, Image<T> &imOut)
    {
	return resize<T>(imIn, size_t(imIn.getWidth()*cx), size_t(imIn.getHeight()*cy), imOut);
    }

    template <class T>
    RES_T scale(Image<T> &imIn, double cx, double cy)
    {
	Image<T> tmpIm(imIn, true); // clone
	return resize(tmpIm, cx, cy, imIn);
    }
    

    template <class T>
    RES_T resize(Image<T> &imIn, UINT sx, UINT sy)
    {
	Image<T> tmpIm(imIn, true); // clone
	imIn.setSize(sx, sy);
	return resize<T>(tmpIm, imIn);
    }


/** @}*/

} // namespace smil


#endif // _D_IMAGE_TRANSFORM_HPP

