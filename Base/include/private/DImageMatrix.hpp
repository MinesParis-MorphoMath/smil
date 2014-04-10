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


#ifndef _D_IMAGE_MATRIX_HPP
#define _D_IMAGE_MATRIX_HPP

#include "DImage.hpp"
#include "DErrors.h"

namespace smil
{
  
    /**
    * \ingroup Base
    * \defgroup Matrix Matrix operations
    * @{
    */

    /**
     * Matrix transposition (for now, only in 2D)
     */
    template <class T>
    RES_T matTrans(const Image<T> &imIn, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn);
	size_t w, h, d;
	imIn.getSize(&w, &h, &d);
	
	ImageFreezer freezer(imOut);
	
// 	typedef typename ImDtTypes<T>::sliceType sliceType;
	typedef typename ImDtTypes<T>::lineType lineType;
	
	if (d==1) // 2D
	{
 	    ASSERT((imOut.setSize(h, w)==RES_OK));
	    
	    lineType pIn = imIn.getPixels();
	    lineType pOut = imOut.getPixels();
	    
	    for (size_t y=0;y<w;y++)
	      for (size_t x=0;x<h;x++,pOut++)
		*pOut = pIn[x*w+y];
		
	    return RES_OK;
	}
	else
	  return RES_ERR_NOT_IMPLEMENTED;
    }
    
    /**
     * Matrix multiplication (for now, only in 2D)
     * 
     * \vectorized
     * \parallelized
     */
    template <class T>
    RES_T matMul(const Image<T> &imIn1, const Image<T> &imIn2, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn1, &imIn2);
	size_t size1[3], size2[3];
	imIn1.getSize(size1);
	imIn2.getSize(size2);
	
 	if (size1[2]!=1 || size2[2]!=1)
	  return RES_ERR_NOT_IMPLEMENTED;
	
	ImageFreezer freezer(imOut);
	
	// Verify that the number of columns m in imIn1 is equal to the number of rows m in imIn2
	ASSERT((size1[0]==size2[1]), "Wrong matrix sizes!", RES_ERR);
	ASSERT((imOut.setSize(size2[0], size1[1])==RES_OK));
	
	Image<T> transIm(size2[1], size2[0]);
	
	// Transpose imIn2 matrix to allow vectorization
	ASSERT((matTrans(imIn2, transIm)==RES_OK));
	
	typedef typename ImDtTypes<T>::sliceType sliceType;
	typedef typename ImDtTypes<T>::lineType lineType;
	
	sliceType lines = imIn1.getLines();
	sliceType outLines = imOut.getLines();
	sliceType cols = transIm.getLines();
	lineType line;
	lineType outLine;
	lineType col;
	
	int nthreads = Core::getInstance()->getNumberOfThreads();
	
	size_t y;
	
	#ifdef USE_OPEN_MP
	      #pragma omp parallel private(line, outLine, col)
	#endif // USE_OPEN_MP
	{
	  
	    #ifdef USE_OPEN_MP
		#pragma omp for schedule(dynamic,nthreads) nowait
	    #endif // USE_OPEN_MP
	    for (y=0;y<size1[1];y++)
	    {
		line = lines[y];
		outLine = outLines[y];
		for (size_t x=0;x<size2[0];x++)
		{
		    col = cols[x];
		    T outVal = 0;
		    for (size_t i=0;i<size1[0];i++)
		      outVal += line[i] * col[i];
		    outLine[x] = outVal;
		    
		}
	    }
	}
	
	return RES_OK;
    }

/** @}*/

} // namespace smil


#endif // _D_IMAGE_MATRIX_HPP

