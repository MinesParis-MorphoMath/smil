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


#ifndef _D_BLOB_MEASURES_HPP
#define _D_BLOB_MEASURES_HPP

#include "Core/include/private/DImage.hpp"
#include "DImageHistogram.hpp"
#include "DMeasures.hpp"

#include <map>

using namespace std;

namespace smil
{
  
    /**
    * \ingroup Base
    * \defgroup BlobMesures Mesures on blobs
    * @{
    */

    /**
    * Measure label areas.
    * Return a map(labelValue, size_t) with the area of each label value.
    * \demo{blob_measures.py}
    */
    template <class T>
    map<UINT, double> measAreas(const Image<T> &imLbl, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measAreaFunc<T> >(imLbl, onlyNonZero);
    }
    
    /**
     * \overload 
     * 
    * Measure label areas from a pre-generated Blob map (faster).
    */
    template <class T>
    map<UINT, double> measAreas(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measAreaFunc<T> >(imIn, blobs);
    }

    /**
    * Measure the minimum value of each blob in imIn.
    * Return a map(labelValue, T) with the min value for each label.
    */
    template <class T>
    map<UINT, T> measMinVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measMinValFunc<T> >(imIn, blobs);
    }

    /**
    * Measure the maximum value of each blob in imIn.
    * Return a map(labelValue, T) with the max value for each label.
    */
    template <class T>
    map<UINT, T> measMaxVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measMaxValFunc<T> >(imIn, blobs);
    }

    /**
    * Measure the min and max values of each blob in imIn.
    * Return a map(labelValue, T) with the min and max values for each label.
    */
    template <class T>
    map<UINT, vector<T> > measRangeVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measMinMaxValFunc<T> >(imIn, blobs);
    }

    /**
    * Measure the mean value and the std dev. of each blob in imIn.
    * Return a map(labelValue, vector<double>) with the mean and std dev. values for each label.
    */
    template <class T>
    map<UINT, Vector_double> measMeanVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measMeanValFunc<T> >(imIn, blobs);
    }

    /**
    * Measure the maximum value of each blob in imIn.
    * Return a map(labelValue, double) with the max value for each label.
    */
    template <class T>
    map<UINT, double> measVolumes(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measVolFunc<T> >(imIn, blobs);
    }
    
    
    /**
    * Measure the list of values of each blob in imIn.
    * Return a map(labelValue, vector<T>) with the list of values for each blob label.
    */
    template <class T>
    map<UINT, vector<T> > valueLists(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, valueListFunc<T> >(imIn, blobs);
    }
        /**
    * Measure the mode value of imIn in each blob.
    * Return a map(labelValue, T) with the m mode value for each label.
    */
    template <class T>
    map<UINT, T > measModeVals(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measModeValFunc<T> >(imIn, blobs);
    }

    /**
    * Measure barycenter of a labeled image.
    * Return a map(labelValue, Point) with the barycenter point coordinates for each label.
    * 
    * \demo{blob_measures.py}
    */
    template <class T>
    map<UINT, Vector_double> measBarycenters(const Image<T> &imLbl, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measBarycenterFunc<T> >(imLbl, onlyNonZero);
    }
    
    /**
    * Measure the barycenter of each blob in imIn.
    * Return a map(labelValue, vector<double>) with the barycenter for each blob label.
    */
    template <class T>
    map<UINT, Vector_double> measBarycenters(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measBarycenterFunc<T> >(imIn, blobs);
    }
    
    

    /**
    * Measure bounding boxes of labeled image.
    * Return a map(labelValue, Box) with the bounding box for each label value.
    */
    template <class T>
    map<UINT, Vector_UINT > measBoundBoxes(const Image<T> &imLbl, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measBoundBoxFunc<T> >(imLbl, onlyNonZero);
    }

    /**
    * Measure bounding boxes of each blob (faster with blobs pre-generated).
    * Return a map(labelValue, Box) with the bounding box for each label value.
    */
    template <class T>
    map<UINT, Vector_UINT > measBoundBoxes(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measBoundBoxFunc<T> >(imIn, blobs);
    }

    /**
    * Measure inertia moments of each label.
    */
    template <class T>
    map<UINT, Vector_double> measInertiaMatrices(const Image<T> &imLbl, const bool onlyNonZero=true)
    {
	return processBlobMeasure<T, measInertiaMatrixFunc<T> >(imLbl, onlyNonZero);
    }
    
    /**
    * Measure blobs inertia moments.
    * 
    * \demo{inertia_moments.py}
    */
    template <class T>
    map<UINT, Vector_double> measInertiaMatrices(const Image<T> &imIn, const map<UINT, Blob> &blobs)
    {
	return processBlobMeasure<T, measInertiaMatrixFunc<T> >(imIn, blobs);
    }
    
/** @}*/

} // namespace smil


#endif // _D_BLOB_MEASURES_HPP

