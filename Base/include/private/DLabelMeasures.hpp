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
#include "DMeasures.hpp"

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
    * \demo{blob_measures.py}
    */
    template <class T>
    map<UINT, double> measAreas(const Image<T> &imIn, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measAreaFunc<T> >(imIn, onlyNonZero);
    }

    template <class T>
    map<UINT, double> measAreas(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measAreaFunc<T> >(imIn, blobs);
    }

    template <class T>
    map<UINT, T> measMinVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measMinValFunc<T> >(imIn, blobs);
    }

    template <class T>
    map<UINT, T> measMaxVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measMaxValFunc<T> >(imIn, blobs);
    }

    template <class T>
    map<UINT, vector<T> > measRangeVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measMinMaxValFunc<T> >(imIn, blobs);
    }

    template <class T>
    map<UINT, DoubleVector> measMeanVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measMeanValFunc<T> >(imIn, blobs);
    }

    template <class T>
    map<UINT, double> measVolumes(const Image<T> &imLbl, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measVolFunc<T> >(imLbl, onlyNonZero);
    }
    
    template <class T>
    map<UINT, double> measVolumes(const Image<T> &imLbl, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measVolFunc<T> >(imLbl, blobs);
    }
    
    /**
    * Measure barycenter of labeled image.
    * Return a map(labelValue, Point) with the barycenter point coordinates for each label value.
    * 
    * \demo{blob_measures.py}
    */
    template <class T>
    map<UINT, DoubleVector> measBarycenters(const Image<T> &imLbl, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measBarycenterFunc<T> >(imLbl, onlyNonZero);
    }
    
    template <class T>
    map<UINT, DoubleVector> measBarycenters(const Image<T> &imLbl, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measBarycenterFunc<T> >(imLbl, blobs);
    }
    
    

    /**
    * Measure bounding boxes of labeled image.
    * Return a map(labelValue, Box) with the bounding box for each label value.
    */
    template <class T>
    map<UINT, UintVector > measBoundBoxes(const Image<T> &imIn, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measBoundBoxFunc<T> >(imIn, onlyNonZero);
    }

    template <class T>
    map<UINT, UintVector > measBoundBoxes(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measBoundBoxFunc<T> >(imIn, blobs);
    }

    template <class T>
    map<UINT, DoubleVector> measInertiaMatrices(const Image<T> &imIn, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measInertiaMatrixFunc<T> >(imIn, onlyNonZero);
    }
    
    template <class T>
    map<UINT, DoubleVector> measInertiaMatrices(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measInertiaMatrixFunc<T> >(imIn, blobs);
    }
    
/** @}*/

} // namespace smil


#endif // _D_LABEL_MEASURES_HPP

