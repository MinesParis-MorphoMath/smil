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
#include <map>

namespace smil
{
  
/**
 * \ingroup Base
 * \defgroup Measures Base measures
 * @{
 */

    /**
    * Volume of an image
    *
    * Returns the sum of the pixel values.
    * \param imIn Input image.
    */
    template <class T>
    size_t vol(const Image<T> &imIn)
    {
	if (!imIn.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;

	size_t npix = imIn.getPixelCount();
	typename ImDtTypes<T>::lineType pixels = imIn.getPixels();
	size_t vol = 0;

	for (int i=0;i<npix;i++)
	    vol += pixels[i];

	return vol;
    }

    /**
    * Min value of an image
    *
    * Returns the min of the pixel values.
    * \param imIn Input image.
    */
    template <class T>
    T minVal(const Image<T> &imIn)
    {
	if (!imIn.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;

	int npix = imIn.getPixelCount();
	typename ImDtTypes<T>::lineType p = imIn.getPixels();
	T minVal = numeric_limits<T>::max();

	for (int i=0;i<npix;i++,p++)
	    if (*p<minVal)
		minVal = *p;

	return minVal;
    }

    /**
    * Max value of an image
    *
    * Returns the min of the pixel values.
    * \param imIn Input image.
    */
    template <class T>
    T maxVal(const Image<T> &imIn)
    {
	if (!imIn.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;

	int npix = imIn.getPixelCount();
	typename ImDtTypes<T>::lineType p = imIn.getPixels();
	T maxVal = numeric_limits<T>::min();

	for (int i=0;i<npix;i++,p++)
	    if (*p>maxVal)
		maxVal = *p;

	return maxVal;
    }

    /**
    * Min and Max values of an image
    *
    * Returns the min and the max of the pixel values.
    * \param imIn Input image.
    */
    template <class T>
    void rangeVal(const Image<T> &imIn, T &ret_min, T &ret_max)
    {
	if (!imIn.isAllocated())
	{
	    ret_min = 0;
	    ret_max = 0;
	    return;
	}

	int npix = imIn.getPixelCount();
	typename ImDtTypes<T>::lineType p = imIn.getPixels();
	ret_min = numeric_limits<T>::max();
	ret_max = numeric_limits<T>::min();

	for (int i=0;i<npix;i++,p++)
	{
	    if (*p<ret_min)
		ret_min = *p;
	    if (*p>ret_max)
		ret_max = *p;
	}

    }

    template <class T>
    void rangeVal(const Image<T> &imIn, T *rVals)
    {
	return rangeVal(imIn, rVals[0], rVals[1]);
    }
    
    template <class T>
    RES_T measBarycenter(Image<T> &im, double *xc, double *yc, double *zc=NULL)
    {
	if (!im.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;
	
	typename Image<T>::volType slices = im.getSlices();
	typename Image<T>::sliceType lines;
	typename Image<T>::lineType pixels;
	T pixVal;
	
	double xSum = 0, ySum = 0, zSum = 0, tSum = 0;
	size_t imSize[3];
	im.getSize(imSize);
	
	for (size_t z=0;z<imSize[2];z++)
	{
	    lines = *slices++;
    // #pragma omp parallel for
	    for (size_t y=0;y<imSize[1];y++)
	    {
		pixels = *lines++;
		for (size_t x=0;x<imSize[0];x++)
		{
		    pixVal = pixels[x];
		    if (pixVal!=0)
		    {
			xSum += pixVal * x;
			ySum += pixVal * y;
			zSum += pixVal * z;
			tSum += pixVal;		  
		    }
		}
	    }
	}
	
	*xc = xSum / tSum;
	*yc = ySum / tSum;
	if (zc)
	  *zc = zSum / tSum;
	
	return RES_OK;
    }

    template <class T>
    vector<double> measBarycenter(Image<T> &im)
    {
	vector<double> res;
	double xc, yc, zc;
	if (measBarycenter<T>(im, &xc, &yc, &zc)==RES_OK)
	{
	    res.push_back(xc);
	    res.push_back(yc);
	    if (im.getDimension()==3)
	      res.push_back(zc);
	}
	return res;
    }

    /**
    * Bounding Box measure
    */
    template <class T>
    RES_T measBoundBox(Image<T> &im, size_t *xMin, size_t *yMin, size_t *zMin, size_t *xMax, size_t *yMax, size_t *zMax)
    {
	if (!im.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;
	
	typename Image<T>::volType slices = im.getSlices();
	typename Image<T>::sliceType lines;
	typename Image<T>::lineType pixels;
    //     T pixVal;
	
	size_t imSize[3];
	im.getSize(imSize);
	
	*xMin = imSize[0];
	*xMax = 0;
	*yMin = imSize[1];
	*yMax = 0;
	*zMin = imSize[2];
	*zMax = 0;
	
	for (size_t z=0;z<imSize[2];z++)
	{
	    lines = *slices++;
	    for (size_t y=0;y<imSize[1];y++)
	    {
		pixels = *lines++;
		for (size_t x=0;x<imSize[0];x++)
		{
		    T pixVal = pixels[x];
		    if (pixVal!=0)
		    {
			if (x<*xMin) *xMin = x;
			else if (x>*xMax) *xMax = x;
			if (y<*yMin) *yMin = y;
			else if (y>*yMax) *yMax = y;
			if (z<*zMin) *zMin = z;
			else if (z>*zMax) *zMax = z;
		    }
		}
	    }
	}
	
	return RES_OK;
    }

    template <class T>
    RES_T measBoundBox(Image<T> &im, size_t *xMin, size_t *yMin, size_t *xMax, size_t *yMax)
    {
	size_t zMin, zMax;
	return measBoundBox(im, xMin, yMin, &zMin, xMax, yMax, &zMax);
    }


    template <class T>
    vector<UINT> measBoundBox(Image<T> &im)
    {
	vector<UINT> res;
	
	size_t b[6];
	UINT dim = im.getDimension()==3 ? 3 : 2;
	
	if (dim==3)
	{
	    if (measBoundBox<T>(im, b, b+1, b+2, b+3, b+4, b+5)!=RES_OK)
	      return res;
	}
	else if (measBoundBox<T>(im, b, b+1, b+2, b+3)!=RES_OK)
	  return res;

	for (UINT i=0;i<dim*2;i++)
	  res.push_back(b[i]);
	
	return res;
    }

    
    /**
    * 2D inertia coefficients
    */
    template <class T>
    RES_T measInertiaCoefficients(Image<T> &im, double *m00, double *m10, double *m01, double *m11, double *m20, double *m02)
    {
	if (!im.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;
	
	typename Image<T>::volType slices = im.getSlices();
	typename Image<T>::sliceType lines = slices[0];
	typename Image<T>::lineType pixels;
	
	size_t imSize[3];
	im.getSize(imSize);
	
	*m00 = *m10 = *m01 = *m11 = *m20 = *m02 = 0.;

	for (size_t y=0;y<imSize[1];y++)
	{
	    pixels = *lines++;
	    for (size_t x=0;x<imSize[0];x++)
	    {
		T pxVal = pixels[x];
		if (pxVal!=0)
		{
		    *m00 += pxVal;
		    *m10 += pxVal * x;
		    *m01 += pxVal * y;
		    *m11 += pxVal * x * y;
		    *m20 += pxVal * x * x;
		    *m02 += pxVal * y * y;
		}
	    }
	}
	return RES_OK;
    }
	
    template <class T>
    vector<double> measInertiaCoefficients(Image<T> &im)
    {
	double m[6];
	measInertiaCoefficients(im, &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
	vector<double> res;
	for (int i=0;i<6;i++)
	  res.push_back(m[i]);
	return res;
    }
    
    /**
    * Non-zero point offsets.
    * Return a vector conatining the offset of all non-zero points in image.
    */
    template <class T>
    vector<size_t> nonZeroOffsets(Image<T> &imIn)
    {
	vector<size_t> offsets;

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

