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


#ifndef _D_MEASURES_HPP
#define _D_MEASURES_HPP

#include "Core/include/private/DImage.hpp"
#include "DBaseMeasureOperations.hpp"
#include <map>

namespace smil
{
  
/**
 * \ingroup Base
 * \defgroup Measures Base measures
 * @{
 */

    template <class T>
    struct measAreaFunc : public MeasureFunctionBase<T, double>
    {
	typedef typename Image<T>::lineType lineType;

	virtual void processSequence(lineType lineIn, size_t size)
	{
	    this->retVal += size;
	}
    };

    /**
    * Area of an image
    *
    * Returns the number of non-zero pixels
    * \param imIn Input image.
    */
    template <class T>
    size_t area(const Image<T> &imIn)
    {
	measAreaFunc<T> func;
	return func(imIn, true);
    }

    
    template <class T>
    struct measVolFunc : public MeasureFunctionBase<T, double>
    {
	typedef typename Image<T>::lineType lineType;

	virtual void processSequence(lineType lineIn, size_t size)
	{
	    for (size_t i=0;i<size;i++)
	      this->retVal += lineIn[i];
	}
    };

    /**
    * Volume of an image
    *
    * Returns the sum of the pixel values.
    * \param imIn Input image.
    */
    template <class T>
    double vol(const Image<T> &imIn)
    {
	measVolFunc<T> func;
	return func(imIn, false);
    }
    
    template <class T>
    struct measMeanValFunc : public MeasureFunctionBase<T, DoubleVector>
    {
	typedef typename Image<T>::lineType lineType;
	double sum1, sum2;
	double pixNbr;

	virtual void initialize(const Image<T> &imIn)
	{
	    this->retVal.clear();
	    sum1 = sum2 = pixNbr = 0.;
	}
	virtual void processSequence(lineType lineIn, size_t size)
	{
	    double curV;
	    for (size_t i=0;i<size;i++)
	    {
		pixNbr += 1;
		curV = lineIn[i];
		sum1 += curV;
		sum2 += curV*curV;
	    }
	}
	virtual void finalize(const Image<T> &imIn)
	{
	    double mean_val = pixNbr==0 ? 0 : sum1/pixNbr;
	    double std_dev_val = pixNbr==0 ? 0 : sqrt(sum2/pixNbr - mean_val*mean_val);
	    
	    this->retVal.push_back(mean_val);
	    this->retVal.push_back(std_dev_val);
	}
    };


    
    /**
    * Mean value and standard deviation
    *
    * Returns mean and standard deviation of the pixel values.
    * If onlyNonZero is true, only non-zero pixels are considered.
    * \param imIn Input image.
    */
    template <class T>
    DoubleVector meanVal(const Image<T> &imIn, bool onlyNonZero=false)
    {
	measMeanValFunc<T> func;
	return func(imIn, onlyNonZero);
    }

    
    template <class T>
    struct measMinValFunc : public MeasureFunctionBase<T, double>
    {
	typedef typename Image<T>::lineType lineType;
	virtual void initialize(const Image<T> &imIn)
	{
	    this->retVal = numeric_limits<T>::max();
	}
	virtual void processSequence(lineType lineIn, size_t size)
	{
	    for (size_t i=0;i<size;i++)
	      if (lineIn[i] < this->retVal)
		this->retVal = lineIn[i];
	}
    };

    /**
    * Min value of an image
    *
    * Returns the min of the pixel values.
    * \param imIn Input image.
    */
    template <class T>
    T minVal(const Image<T> &imIn, bool onlyNonZero=false)
    {
	measMinValFunc<T> func;
	return func(imIn, onlyNonZero);
    }

    template <class T>
    struct measMaxValFunc : public MeasureFunctionBase<T, double>
    {
	typedef typename Image<T>::lineType lineType;
	virtual void initialize(const Image<T> &imIn)
	{
	    this->retVal = numeric_limits<T>::min();
	}
	virtual void processSequence(lineType lineIn, size_t size)
	{
	    for (size_t i=0;i<size;i++)
	      if (lineIn[i] > this->retVal)
		this->retVal = lineIn[i];
	}
    };

    /**
    * Max value of an image
    *
    * Returns the min of the pixel values.
    * \param imIn Input image.
    */
    template <class T>
    T maxVal(const Image<T> &imIn, bool onlyNonZero=false)
    {
	measMaxValFunc<T> func;
	return func(imIn, onlyNonZero);
    }

    
    template <class T>
    struct measMinMaxValFunc : public MeasureFunctionBase<T, DoubleVector>
    {
	typedef typename Image<T>::lineType lineType;
	double minVal, maxVal;
	virtual void initialize(const Image<T> &imIn)
	{
	    this->retVal.clear();
	    maxVal = numeric_limits<T>::min();
	    minVal = numeric_limits<T>::max();
	}
	virtual void processSequence(lineType lineIn, size_t size)
	{
	    for (size_t i=0;i<size;i++)
	    {
	      T val = lineIn[i];
	      if (val > maxVal)
		maxVal = val;
	      if (val < minVal)
		minVal = val;
	    }
	}
	virtual void finalize(const Image<T> &imIn)
	{
	    this->retVal.push_back(minVal);
	    this->retVal.push_back(maxVal);
	}
    };
    /**
    * Min and Max values of an image
    *
    * Returns the min and the max of the pixel values.
    * \param imIn Input image.
    */
    template <class T>
    vector<T> rangeVal(const Image<T> &imIn, bool onlyNonZero=false)
    {
	measMinMaxValFunc<T> func;
	DoubleVector dVec = func(imIn, onlyNonZero);
	return vector<T>(dVec.begin(), dVec.end());
    }

    
    template <class T>
    struct measBarycenterFunc : public MeasureFunctionWithPos<T, DoubleVector>
    {
	typedef typename Image<T>::lineType lineType;
	double xSum, ySum, zSum, tSum;
	virtual void initialize(const Image<T> &imIn)
	{
	    this->retVal.clear();
	    xSum = ySum = zSum = tSum = 0.;
	}
	virtual void processSequence(lineType lineIn, size_t size, size_t x, size_t y, size_t z)
	{
	    for (size_t i=0;i<size;i++,x++)
	    {
	      T pixVal = lineIn[i];
	      xSum += pixVal * x;
	      ySum += pixVal * y;
	      zSum += pixVal * z;
	      tSum += pixVal;		  
	    }
	}
	virtual void finalize(const Image<T> &imIn)
	{
	    this->retVal.push_back(xSum/tSum);
	    this->retVal.push_back(ySum/tSum);
	    if (imIn.getDimension()==3)
	      this->retVal.push_back(zSum/tSum);
	}
    };
    
    template <class T>
    DoubleVector measBarycenter(Image<T> &im)
    {
	measBarycenterFunc<T> func;
	return func(im, false);
    }


    template <class T>
    struct measBoundBoxFunc : public MeasureFunctionWithPos<T, DoubleVector>
    {
	typedef typename Image<T>::lineType lineType;
	double xMin, xMax, yMin, yMax, zMin, zMax;
	virtual void initialize(const Image<T> &imIn)
	{
	    this->retVal.clear();
	    size_t imSize[3];
	    imIn.getSize(imSize);
	    
	    xMin = imSize[0];
	    xMax = 0;
	    yMin = imSize[1];
	    yMax = 0;
	    zMin = imSize[2];
	    zMax = 0;
	}
	virtual void processSequence(lineType lineIn, size_t size, size_t x, size_t y, size_t z)
	{
	    if (x<xMin) xMin = x;
	    if (x+size-1>xMax) xMax = x+size-1;
	    if (y<yMin) yMin = y;
	    if (y>yMax) yMax = y;
	    if (z<zMin) zMin = z;
	    if (z>zMax) zMax = z;
	}
	virtual void finalize(const Image<T> &imIn)
	{
	    this->retVal.push_back(xMin);
	    this->retVal.push_back(xMax);
	    this->retVal.push_back(yMin);
	    this->retVal.push_back(yMax);
	    if (imIn.getDimension()==3)
	    {
	      this->retVal.push_back(zMin);
	      this->retVal.push_back(zMax);
	    }
	}
    };
    /**
    * Bounding Box measure
    */
    template <class T>
    vector<UINT> measBoundBox(Image<T> &im)
    {
	measBoundBoxFunc<T> func;
	DoubleVector dVec = func(im, true);
	return vector<UINT>(dVec.begin(), dVec.end());
    }


    template <class T>
    struct measInertiaMatrixFunc : public MeasureFunctionWithPos<T, DoubleVector>
    {
	typedef typename Image<T>::lineType lineType;
	double m00, m10, m01, m11, m20, m02;
	virtual void initialize(const Image<T> &imIn)
	{
	    this->retVal.clear();
	    m00 = m10 = m01 = m11 = m20 = m02 = 0.;
	}
	virtual void processSequence(lineType lineIn, size_t size, size_t x, size_t y, size_t z)
	{
	    for (size_t x=0;x<size;x++)
	    {
		T pxVal = lineIn[x];
		m00 += pxVal;
		m10 += pxVal * x;
		m01 += pxVal * y;
		m11 += pxVal * x * y;
		m20 += pxVal * x * x;
		m02 += pxVal * y * y;
	    }
	}
	virtual void finalize(const Image<T> &imIn)
	{
	    this->retVal.push_back(m00);
	    this->retVal.push_back(m10);
	    this->retVal.push_back(m01);
	    this->retVal.push_back(m11);
	    this->retVal.push_back(m20);
	    this->retVal.push_back(m02);
	}
    };
    /**
    * 2D inertia coefficients
    */
    template <class T>
    DoubleVector measInertiaMatrix(const Image<T> &im, const bool onlyNonZero=true)
    {
	measInertiaMatrixFunc<T> func;
	return func(im, onlyNonZero);
    }
	
    /**
    * Non-zero point offsets.
    * Return a vector conatining the offset of all non-zero points in image.
    */
    template <class T>
    vector<UINT> nonZeroOffsets(Image<T> &imIn)
    {
	vector<UINT> offsets;

	ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, offsets);
	
	typename Image<T>::lineType pixels = imIn.getPixels();
	
	for (size_t i=0;i<imIn.getPixelCount();i++)
	  if (pixels[i]!=0)
	    offsets.push_back(i);
    
	return offsets;
    }

/** @}*/

} // namespace smil


#endif // _D_MEASURES_HPP

