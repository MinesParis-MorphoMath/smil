/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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
#include "Core/include/private/DBufferPool.hpp"


namespace smil
{
  
    /**
    * \ingroup Base
    * \defgroup Convolution
    * @{
    */
    

    
    /**
     * Horizontal convolution
     * 
     * \b Example:
     * \code{.py}
     * im1 = Image("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
     * im2 = Image(im1)
     * kern = [ 0.0545, 0.2442, 0.4026, 0.2442, 0.0545 ]
     * horizConvolve(im1, kern, im2)
     * showAll()
     * \endcode
     */
    // Inplace safe
    template <class T>
    RES_T horizConvolve(const Image<T> &imIn, const vector<double> &kernel, Image<T> &imOut)
    {
	CHECK_ALLOCATED(&imIn, &imOut);
	CHECK_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	typename ImDtTypes<T>::sliceType linesIn = imIn.getLines();
	typename ImDtTypes<T>::sliceType linesOut = imOut.getLines();
	typename ImDtTypes<T>::lineType lOut;
	
	int imW = imIn.getWidth();
	int kernelRadius = (kernel.size()-1)/2;
// 	int kLen = 2*kernelRadius+1;
	
	double *partialKernWeights = new double[kernelRadius];
	double pkwSum = 0;
	for (int i=0;i<kernelRadius;i++)
	  pkwSum += kernel[i];
	for (int i=0;i<kernelRadius;i++)
	{
	    pkwSum += kernel[i+kernelRadius];
	    partialKernWeights[i] = pkwSum;
	}
	
	typedef double bufType; // If float, loops are vectorized
	BufferPool<bufType> bufferPool(imW);
	  
	#ifdef USE_OPEN_MP
	      int nthreads = Core::getInstance()->getNumberOfThreads();
	      #pragma omp parallel private(lOut) num_threads(nthreads)
	#endif // USE_OPEN_MP
	{	
	    typename ImDtTypes<bufType>::lineType lIn = bufferPool.getBuffer();
	    double sum;
	    
	    #ifdef USE_OPEN_MP
		#pragma omp for
	    #endif // USE_OPEN_MP
	    for (int y=0;y<(int)imIn.getLineCount();y++)
	    {
		copyLine<T,bufType>(linesIn[y], imW, lIn);
		lOut = linesOut[y];
		
		// left pixels
		for (int x=0;x<kernelRadius;x++)
		{
		  sum = 0;
		  for (int i=-x;i<=kernelRadius;i++)
		      sum += kernel[i+kernelRadius]*lIn[x+i];
		  lOut[x] = T(sum / partialKernWeights[x]);
		    
		}
		
		// center pixels
		for (int x=kernelRadius;x<imW-kernelRadius;x++)
		{
		  sum = 0;
		  for (int i=-kernelRadius;i<=kernelRadius;i++)
		      sum += kernel[i+kernelRadius]*lIn[x+i];
		  lOut[x] = T(sum);
		}
		
		// right pixels
		for (int x=imW-kernelRadius;x<imW;x++)
		{
		  sum = 0;
		  for (int i=-kernelRadius;i<imW-x;i++)
		      sum += kernel[i+kernelRadius]*lIn[x+i];
		  lOut[x] = T(sum / partialKernWeights[imW-1-x]);
		}
	    }
	    
	}
	delete[] partialKernWeights;
	return RES_OK;
    }
    
    /**
     * Vertical convolution
     * 
     * \see horizConvolve
     */
    template <class T>
    RES_T vertConvolve(const Image<T> &imIn, const vector<double> &kernel, Image<T> &imOut)
    {
	CHECK_ALLOCATED(&imIn, &imOut);
	CHECK_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	typename ImDtTypes<T>::volType slicesIn = imIn.getSlices();
	typename ImDtTypes<T>::volType slicesOut = imOut.getSlices();
	typename ImDtTypes<T>::sliceType sIn, sOut;
	
	int imW = imIn.getWidth();
	int imH = imIn.getHeight();
	int imD = imIn.getDepth();
	int kernelRadius = (kernel.size()-1)/2;
	int nthreads = Core::getInstance()->getNumberOfThreads();
	
	double *partialKernWeights = new double[kernelRadius];
	double pkwSum = 0;
	for (int i=0;i<kernelRadius;i++)
	  pkwSum += kernel[i];
	for (int i=0;i<kernelRadius;i++)
	{
	    pkwSum += kernel[i+kernelRadius];
	    partialKernWeights[i] = pkwSum;
	}
	  
	typedef double bufType; // If double, loops are vectorized
	BufferPool<bufType> bufferPool(imW);
	
	for (int z=0;z<imD;z++)
	{
	    #ifdef USE_OPEN_MP
		  #pragma omp parallel private(sIn, sOut) num_threads(nthreads)
	    #endif // USE_OPEN_MP
	    {
		sIn = slicesIn[z];
		sOut = slicesOut[z];
		double sum;
		
		#ifdef USE_OPEN_MP
		    #pragma omp for
		#endif // USE_OPEN_MP
		for (int x=0;x<imW;x++)
		{
		    // Top pixels
		    for (int y=0;y<kernelRadius;y++)
		    {
		      sum = 0;
		      for (int i=-y;i<kernelRadius+1;i++)
			sum += kernel[i+kernelRadius]*sIn[y+i][x];
		      sOut[y][x] = T(sum / partialKernWeights[y]);
		    }
		    
		    // Center pixels
		    for (int y=kernelRadius;y<imH-kernelRadius;y++)
		    {
		      sum = 0;
		      for (int i=-kernelRadius;i<=kernelRadius;i++)
			sum += kernel[i+kernelRadius]*sIn[y+i][x];
		      sOut[y][x] = T(sum);
		    }
		    
		    // Bottom pixels
		    for (int y=imH-kernelRadius;y<imH;y++)
		    {
		      sum = 0;
		      for (int i=-kernelRadius;i<imH-y;i++)
			sum += kernel[i+kernelRadius]*sIn[y+i][x];
		      sOut[y][x] = T(sum / partialKernWeights[imH-1-y]);
		    }
		}
	    }
	}
	delete[] partialKernWeights;
	return RES_OK;
    }
    
    /**
     * Convolution in both x and y directions using the same 1D kernel
     * 
     * \see horizConvolve
     */
    template <class T>
    RES_T convolve(const Image<T> &imIn, const vector<double> &kernel, Image<T> &imOut)
    {
	if (&imIn==&imOut)
	{
	    Image<T> tmpIm(imIn, true); // clone
	    return convolve(tmpIm, kernel, imOut);
	}
	
	CHECK_ALLOCATED(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	vertConvolve(imIn, kernel, imOut);
	horizConvolve(imOut, kernel, imOut); // inplace safe
	
	return RES_OK;
    }
    
    
    /**
     * 2D Gaussian filter
     * 
     * The size of the filter is 2*radius+1
     */
    template <class T>
    RES_T gaussianFilter(const Image<T> &imIn, size_t radius, Image<T> &imOut)
    {
	CHECK_ALLOCATED(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	int kernelSize = radius*2+1;
	vector<double> kernel(kernelSize);
	
	double sigma = double(radius)/2.;
	double sum = 0.;
	
	//  Determine kernel coefficients
	for (int i=0;i<kernelSize;i++)
	{
	  kernel[i] = double(exp( -pow((double(i)-radius)/sigma, 2)/2. ));
	  sum += kernel[i];
	}
	// Normalize
	for (int i=0;i<kernelSize;i++)
	  kernel[i] /= sum;


	return convolve(imIn, kernel, imOut);
	
    }
    

/** @}*/

} // namespace smil


#endif // _D_IMAGE_DRAW_HPP

