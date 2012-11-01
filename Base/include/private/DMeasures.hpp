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

/**
 * \ingroup Base
 * \defgroup Measures Base measures
 * @{
 */

#include "DImage.hpp"
#include <map>


/**
 * Volume of an image
 *
 * Returns the sum of the pixel values.
 * \param imIn Input image.
 */
template <class T>
double vol(const Image<T> &imIn)
{
    if (!imIn.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    int npix = imIn.getPixelCount();
    typename ImDtTypes<T>::lineType pixels = imIn.getPixels();
    double vol = 0;

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
void rangeVal(const Image<T> &imIn, T *ret_min, T *ret_max)
{
    if (!imIn.isAllocated())
    {
	*ret_min = 0;
	*ret_max = 0;
        return;
    }

    int npix = imIn.getPixelCount();
    typename ImDtTypes<T>::lineType p = imIn.getPixels();
    *ret_min = numeric_limits<T>::max();
    *ret_max = numeric_limits<T>::min();

    for (int i=0;i<npix;i++,p++)
    {
        if (*p<*ret_min)
            *ret_min = *p;
        if (*p>*ret_max)
            *ret_max = *p;
    }

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
    UINT imSize[3];
    im.getSize(imSize);
    
    for (UINT z=0;z<imSize[2];z++)
    {
	lines = *slices++;
// #pragma omp parallel for
	for (UINT y=0;y<imSize[1];y++)
	{
	    pixels = *lines++;
	    for (UINT x=0;x<imSize[0];x++)
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
RES_T measBoundBox(Image<T> &im, UINT *xMin, UINT *yMin, UINT *zMin, UINT *xMax, UINT *yMax, UINT *zMax)
{
    if (!im.isAllocated())
        return RES_ERR_BAD_ALLOCATION;
    
    typename Image<T>::volType slices = im.getSlices();
    typename Image<T>::sliceType lines;
    typename Image<T>::lineType pixels;
//     T pixVal;
    
    UINT imSize[3];
    im.getSize(imSize);
    
    *xMin = imSize[0];
    *xMax = 0;
    *yMin = imSize[1];
    *yMax = 0;
    *zMin = imSize[2];
    *zMax = 0;
    
    for (UINT z=0;z<imSize[2];z++)
    {
	lines = *slices++;
	for (UINT y=0;y<imSize[1];y++)
	{
	    pixels = *lines++;
	    for (UINT x=0;x<imSize[0];x++)
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
RES_T measBoundBox(Image<T> &im, UINT *xMin, UINT *yMin, UINT *xMax, UINT *yMax)
{
    UINT zMin, zMax;
    return measBoundBox(im, xMin, yMin, &zMin, xMax, yMax, &zMax);
}


template <class T>
vector<UINT> measBoundBox(Image<T> &im)
{
    vector<UINT> res;
    
    UINT b[6];
    UINT dim = im.getDimension()==3 ? 3 : 2;
    
    if (dim==3 && measBoundBox<T>(im, b, b+1, b+2, b+3, b+4, b+5)!=RES_OK)
      return res;
    else if (measBoundBox<T>(im, b, b+1, b+2, b+3)!=RES_OK)
      return res;

    for (UINT i=0;i<dim*2;i++)
      res.push_back(b[i]);
    
    return res;
}


/** @}*/

#endif // _D_MEASURES_HPP

