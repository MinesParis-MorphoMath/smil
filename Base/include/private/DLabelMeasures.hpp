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


#ifndef _D_LABEL_MEASURES_HPP
#define _D_LABEL_MEASURES_HPP

#include "Core/include/private/DImage.hpp"
#include "DImageHistogram.hpp"

#include <map>

using namespace std;

namespace smil
{
  
    /**
    * \ingroup Measures
    * @{
    */


    /**
    * Measure label areas.
    * Return a map(labelValue, size_t) with the area of each label value.
    */
    template <class T>
    map<T, double> measAreas(Image<T> &imIn)
    {
	map<T, double> area;

	ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, area);
	
	typename Image<T>::volType slices = imIn.getSlices();
	typename Image<T>::sliceType lines;
	typename Image<T>::lineType pixels;
	T pixVal;
	
	size_t imSize[3];
	imIn.getSize(imSize);
	
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
			area[pixVal] += 1;
		}
	    }
	}
	
	return area;
    }



    /**
    * Measure barycenter of labeled image.
    * Return a map(labelValue, Point) with the barycenter point coordinates for each label value.
    */
    template <class T>
    map<T, DoublePoint> measBarycenters(Image<T> &imIn)
    {
	map<T, DoublePoint> res;
	
	ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, res);
	
	typename Image<T>::volType slices = imIn.getSlices();
	typename Image<T>::sliceType lines;
	typename Image<T>::lineType pixels;
	T pixVal;
	
	map<T, double> xc, yc, zc;
	map<T, UINT> ptNbrs;
	
	size_t imSize[3];
	imIn.getSize(imSize);
	
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
			xc[pixVal] += x;
			yc[pixVal] += y;
			zc[pixVal] += z;
			ptNbrs[pixVal]++;
		    }
		}
	    }
	}
	
	typename map<T, UINT>::iterator it;
	
	for ( it=ptNbrs.begin() ; it != ptNbrs.end(); it++ )
	{
	    T lblVal = (*it).first;
	    DoublePoint p;
	    p.x = xc[lblVal]/ptNbrs[lblVal];
	    p.y = yc[lblVal]/ptNbrs[lblVal];
	    p.z = zc[lblVal]/ptNbrs[lblVal];
	    res[lblVal] = p;
	}
	
	return res;
    }
    
/** @}*/

} // namespace smil


#endif // _D_LABEL_MEASURES_HPP

