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


#ifndef _D_IMAGE_CONVOLUTION_HPP
#define _D_IMAGE_CONVOLUTION_HPP

#include "DLineArith.hpp"


namespace smil
{
  
    /**
    * \ingroup Base
    * \defgroup Convolution
    * @{
    */
    

    template <class T1, class T2>
    RES_T convolve(const Image<T1> &imIn, const Image<T2> &imKernel, Image<T1> &imOut)
    {
	CHECK_ALLOCATED(&imIn, &imKernel, &imOut);
	
	ImageFreezer freeze(imOut);
    }
    
    template <class T>
    RES_T horizConvolve(const Image<T> &imIn, const float *kernel, int kernelRadius, Image<T> &imOut)
    {
	CHECK_ALLOCATED(&imIn, &imOut);
	CHECK_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	typename ImDtTypes<T>::sliceType linesIn = imIn.getLines();
	typename ImDtTypes<T>::sliceType linesOut = imOut.getLines();
	typename ImDtTypes<T>::lineType lIn, lOut;
	
	size_t imW = imIn.getWidth();
	int kLen = 2*kernelRadius+1;
	
	#ifdef USE_OPEN_MP
	      int nthreads = Core::getInstance()->getNumberOfThreads();
	      #pragma omp parallel private(lIn, lOut) num_threads(nthreads)
	#endif // USE_OPEN_MP
	{	
	    #ifdef USE_OPEN_MP
		#pragma omp for
	    #endif // USE_OPEN_MP
	    for (int y=0;y<imIn.getLineCount();y++)
	    {
		lIn = linesIn[y];
		lOut = linesOut[y];
		for (int x=0;x<imW;x++)
		{
		  float sum = 0;
		  for (int i=-kernelRadius;i<=kernelRadius;i++)
		    if (x>=-i && x+i<imW)
		      sum += kernel[i+kernelRadius]*lIn[x+i];
		  lOut[x] = sum;
		}
	    }
	}
	return RES_OK;
    }
    
    template <class T>
    RES_T vertConvolve(const Image<T> &imIn, const float *kernel, int kernelRadius, Image<T> &imOut)
    {
	CHECK_ALLOCATED(&imIn, &imOut);
	CHECK_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	typename ImDtTypes<T>::volType slicesIn = imIn.getSlices();
	typename ImDtTypes<T>::volType slicesOut = imOut.getSlices();
	typename ImDtTypes<T>::sliceType sIn, sOut;
	typename ImDtTypes<T>::lineType lIn, lOut;
	
	size_t imW = imIn.getWidth();
	size_t imH = imIn.getHeight();
	size_t imD = imIn.getDepth();
	int kLen = 2*kernelRadius+1;
	int nthreads = Core::getInstance()->getNumberOfThreads();
	
	for (int z=0;z<imD;z++)
	{
	    sIn = slicesIn[z];
	    sOut = slicesOut[z];
	
	    #ifdef USE_OPEN_MP
		  #pragma omp parallel firstprivate(sIn, sOut) num_threads(nthreads)
	    #endif // USE_OPEN_MP
	    {
		#ifdef USE_OPEN_MP
		    #pragma omp for
		#endif // USE_OPEN_MP
		for (int x=0;x<imW;x++)
		{
		    for (int y=0;y<imH;y++)
		    {
		      float sum = 0;
		      for (int i=-kernelRadius;i<=kernelRadius;i++)
			if (y>=-i && y+i<imH)
			  sum += kernel[i+kernelRadius]*sIn[y+i][x];
		      sOut[y][x] = sum;
		    }
		}
	    }
	}
	return RES_OK;
    }
    
    template <class T>
    RES_T gaussianFilter(const Image<T> &imIn, size_t radius, Image<T> &imOut)
    {
	CHECK_ALLOCATED(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	int kernSize = radius*2+1;
	float *kern = ImDtTypes<float>::createLine(kernSize);
// 	float kern[] = { 0.0545, 0.2442, 0.4026, 0.2442, 0.0545 };
	
	float sigma = float(radius)/2.;
	float sum = 0.;
	
	//  Determine kernel coefficients
	for (int i=0;i<kernSize;i++)
	{
	  kern[i] = exp( -pow((float(i)-radius)/sigma, 2)/2. );
	  sum += kern[i];
	}
	// Normalize
	for (int i=0;i<kernSize;i++)
	  kern[i] /= 2*sum;

	size_t imW = imIn.getWidth();
	size_t imH = imIn.getHeight();
	
	Image<T> tmpIm(imIn);
	
	typename Image<T>::sliceType lines = imIn.getLines();
	horizConvolve(imIn, kern, radius, tmpIm);
	vertConvolve(tmpIm, kern, radius, imOut);
	
	return RES_OK;
    }
    

/** @}*/

} // namespace smil


#endif // _D_IMAGE_DRAW_HPP

