/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
   * @ingroup Base
   * @defgroup BlobMesures Mesures on blobs
   * @{
   */

  /**
   * Calculate the area of each region in a labeled image
   *
   * @param[in] imLbl : input labeled image
   * @param[in] onlyNonZero : skip a blob having a null label
   * @return a map of pairs <b><label, double></b> with the @b area of each blob.
   *
   * @smilexample{blob_measures.py}
   */
  template <class T>
  map<T, double> measAreas(const Image<T> &imLbl, const bool onlyNonZero = true)
  {
    return processBlobMeasure<T, T, measAreaFunc<T>>(imLbl, onlyNonZero);
  }

  /**
   *
   * Measure areas from a pre-generated Blob map (faster).
   *
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, double></b> with the @b area of each blob.
   *
   */
  template <class labelT>
  map<labelT, double> measAreas(map<labelT, Blob> &blobs)
  {
    Image<labelT> fakeImg(1, 1);
    return processBlobMeasure<labelT, labelT, measAreaFunc<labelT>>(fakeImg,
                                                                    blobs);
  }

  /**
   * Measure the minimum value of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, T></b> with the @b min image value in each blob.
   */
  template <class T, class labelT>
  map<labelT, T> measMinVals(const Image<T> &imIn, map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMinValFunc<T>>(imIn, blobs);
  }

  /**
   * Measure the maximum value of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, T></b> with the @b max image value in each blob.
   */
  template <class T, class labelT>
  map<labelT, T> measMaxVals(const Image<T> &imIn, map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMaxValFunc<T>>(imIn, blobs);
  }

  /**
   * Measure the min and max values of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, <T, T>></b> with a vector with the @b min and @b max pixel values in each blob.
   */
  template <class T, class labelT>
  map<labelT, vector<T>> measRangeVals(const Image<T> &imIn,
                                       map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMinMaxValFunc<T>>(imIn, blobs);
  }

  /**
   * Measure the mean value and the std dev. of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, <double, double>></b> with a vector with the @b mean and <b> standard deviation</b> pixel values in each blob.
   */
  template <class T, class labelT>
  map<labelT, Vector_double> measMeanVals(const Image<T> &imIn,
                                          map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMeanValFunc<T>>(imIn, blobs);
  }

  /**
   * Measure the sum of values of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, double></b> with the @b volume (sum of pixel values) in each blob.
   */
  template <class T, class labelT>
  map<labelT, double> measVolumes(const Image<T> &imIn,
                                  map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measVolFunc<T>>(imIn, blobs);
  }

  /**
   * Measure the list of values of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, vector<T>></b> with the list of pixel values in each blob.
   */
  template <class T, class labelT>
  map<labelT, vector<T>> valueLists(const Image<T> &imIn,
                                    map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, valueListFunc<T>>(imIn, blobs);
  }
  /**
   * Measure the mode value of imIn in each blob.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, T></b> with the @b mode value in each blob.
   */
  template <class T, class labelT>
  map<labelT, T> measModeVals(const Image<T> &imIn, map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measModeValFunc<T>>(imIn, blobs);
  }
  /**
   * Measure the median value of imIn in each blob.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, T></b> with the @b median value in each blob.
   */
  template <class T, class labelT>
  map<labelT, T> measMedianVals(const Image<T> &imIn, map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMedianValFunc<T>>(imIn, blobs);
  }

  /**
   * Measure barycenter of a labeled image.
   *
   * @param[in] imLbl : input labeled image
   * @param[in] onlyNonZero : skip a blob having a null label
   * @return a map of pairs <b><label, Vector<double>></b> with the coordinates of the @b barycenter of each blob.
   *
   * @smilexample{blob_measures.py}
   */
  template <class T>
  map<T, Vector_double> measBarycenters(const Image<T> &imLbl,
                                        const bool onlyNonZero = true)
  {
    return processBlobMeasure<T, T, measBarycenterFunc<T>>(imLbl, onlyNonZero);
  }

  /**
   * Measure the barycenter of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, Vector<double>></b> with the coordinates of the @b barycenter of each blob.
   */
  template <class T, class labelT>
  map<labelT, Vector_double> measBarycenters(const Image<T> &imIn,
                                             map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measBarycenterFunc<T>>(imIn, blobs);
  }

  /**
   * Measure bounding boxes of labeled image.
   *
   * @param[in] imLbl : input labeled image
   * @param[in] onlyNonZero : skip a blob having a null label
   * @return a map of pairs <b><label, Vector<size_t>></b> with the coordinates of the <b>bounding box</b> of each label.
   */
  template <class T>
  map<T, vector<size_t>> measBoundBoxes(const Image<T> &imLbl,
                                        const bool onlyNonZero = true)
  {
    return processBlobMeasure<T, T, measBoundBoxFunc<T>>(imLbl, onlyNonZero);
  }

  /**
   * Measure bounding boxes of each blob (faster with blobs pre-generated).
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, Vector<size_t>></b> with the coordinates of the <b>bounding box</b> of each label.
   */
  template <class T, class labelT>
  map<labelT, vector<size_t>> measBoundBoxes(const Image<T> &imIn,
                                             map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measBoundBoxFunc<T>>(imIn, blobs);
  }

  /**
   * Measure image moments of each label.
   *
   * @param[in] imLbl : input labeled image
   * @param[in] onlyNonZero : skip a blob having a null label
   * @return a map of pairs <b><label, Vector<double>></b> with the <b>image moments</b> of each label.
   * @see measImageMoments()
   */
  template <class T>
  map<T, Vector_double> measImageBlobsMoments(const Image<T> &imLbl,
                                              const bool onlyNonZero = true)
  {
    return processBlobMeasure<T, T, measImageMomentsFunc<T>>(imLbl,
                                                             onlyNonZero);
  }

  /**
   * Measure blobs image moments (faster with blobs pre-generated).
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, Vector<double>></b> with the <b>image moments</b> of each label..
   * @see measImageMoments()
   *
   * @smilexample{inertia_moments.py}
   */
  template <class T, class labelT>
  map<labelT, Vector_double> measImageBlobsMoments(const Image<T> &imIn,
                                                   map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measImageMomentsFunc<T>>(imIn, blobs);
  }

  /** @}*/

} // namespace smil

#endif // _D_BLOB_MEASURES_HPP
