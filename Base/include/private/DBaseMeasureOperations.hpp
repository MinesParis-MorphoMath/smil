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


#ifndef _D_BASE_MEASURE_OPERATIONS_HPP
#define _D_BASE_MEASURE_OPERATIONS_HPP

#include "Core/include/private/DImage.hpp"
#include <map>

namespace smil
{
  
/**
 * \ingroup Base
 * \defgroup Measures Base measures
 * @{
 */

    struct PixelSequence
    {
	size_t offset;
	size_t size;
	PixelSequence() : offset(0), size(0) {}
	PixelSequence(size_t off, size_t siz) : offset(off), size(siz) {}
    };
    
    /**
     * List of offset and size of (memory) contiguous pixels
     */
    struct Blob
    {
      vector<PixelSequence> sequences;
      typedef typename vector<PixelSequence>::iterator sequences_iterator;
      typedef typename vector<PixelSequence>::const_iterator sequences_const_iterator;
    };
    
    template <class T>
    map<T, Blob> computeBlobs(const Image<T> &imIn, bool onlyNonZero=true)
    {
	map<T, Blob> blobs;
	
	ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, blobs);

	size_t npix = imIn.getPixelCount();
	typename ImDtTypes<T>::lineType pixels = imIn.getPixels();
	size_t curSize = 0;
	size_t curStart = 0;
	
	T curVal = pixels[0];
	if (curVal!=0 || !onlyNonZero)
	  curSize++;
	
	for (size_t i=1;i<npix;i++)
	{
	    if (pixels[i]==curVal)
	      curSize++;
	    else
	    {
	      if (curVal!=0 || !onlyNonZero)
		blobs[curVal].sequences.push_back(PixelSequence(curStart, curSize));
	      curStart = i;
	      curSize = 1;
	      curVal = pixels[i];
	    }
	}
	if (curVal!=0 || !onlyNonZero)
	  blobs[curVal].sequences.push_back(PixelSequence(curStart, curSize));
	
	return blobs;
    }

    template <class retType>
    inline void initialize_ret_type(retType &retVal)
    {
	retVal = 0;
    }
    template <>
    inline void initialize_ret_type<DoubleVector>(DoubleVector &retVal)
    {
	retVal.clear();
    }
    
    
    template <class T, class _retType>
    struct MeasureFunctionBase
    {
	typedef typename Image<T>::lineType lineType;
	typedef _retType retType;
	retType retVal;
	
	virtual void initialize()
	{
	    initialize_ret_type(retVal);
	}
	virtual void processSequence(lineType lineIn, size_t size) {}
	virtual void finalize() {}
	
	virtual RES_T processImage(const Image<T> &imIn, bool onlyNonZero=false)
	{
	    initialize();
	    ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION);
	    
	    lineType pixels = imIn.getPixels();
	    size_t pixCount = imIn.getPixelCount();
	    
	    if (!onlyNonZero)
		processSequence(pixels, pixCount);
	    else
	    {
		size_t curSize = 0;
		size_t curStart = 0;
		
		for (size_t i=0;i<pixCount;i++)
		{
		    if (pixels[i]!=0)
		      curSize++;
		    else if (curSize>0)
		    {
		      processSequence(pixels + curStart, curSize);
		      curStart = i;
		      curSize = 0;
		    }
		}
		if (curSize>0)
		  processSequence(pixels + curStart, curSize);
	    }
	    finalize();
	    return RES_OK;
	}
	virtual retType operator()(const Image<T> &imIn, bool onlyNonZero=false)
	{
	    processImage(imIn, onlyNonZero);
	    return retVal;
	}
	
	virtual retType processImage(const Image<T> &imIn, const Blob &blob)
	{
	    initialize();
	    
	    ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, retVal);
	    
	    lineType pixels = imIn.getPixels();
	    Blob::sequences_const_iterator it = blob.sequences.begin();
	    Blob::sequences_const_iterator it_end = blob.sequences.end();
	    for (;it!=it_end;it++)
	      processSequence(pixels + (*it).offset, (*it).size);
	    finalize();
	    return retVal;
	}
	virtual retType operator()(const Image<T> &imIn, const Blob &blob)
	{
	    processImage(imIn, blob);
	    return retVal;
	}
	
    };
    
    
    
    
    template <class T, class funcT>
    map<T, typename funcT::retType> processBlobMeasure(const Image<T> &imIn, const map<T, Blob> &blobs)
    {
	map<T, typename funcT::retType> res;
	
	ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, res);
	
	typename map<T, Blob>::const_iterator it;
	for (it=blobs.begin();it!=blobs.end();it++)
	{
	    funcT func;
	    res[it->first] = func(imIn, it->second);
	}
	return res;
    }
    
    template <class T, class funcT>
    map<T, typename funcT::retType> processBlobMeasure(const Image<T> &imIn, bool onlyNonZero=true)
    {
	map<T, Blob> blobs = computeBlobs(imIn, onlyNonZero);
	return processBlobMeasure<T,funcT>(imIn, blobs);
    }


/** @}*/

} // namespace smil


#endif // _D_BASE_MEASURE_OPERATIONS_HPP

