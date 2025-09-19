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

namespace smil
{
  /**
   * @ingroup Base
   * @defgroup BlobMesures Mesures on blobs
   * @{
   */

  /**
   * blobsArea() - Calculate the area of each region in a labeled image
   *
   * @param[in] imLbl : input labeled image
   * @param[in] onlyNonZero : skip a blob having a null label
   * @return a map of pairs <b><label, double></b> with the @b area of each
   * blob.
   *
   * @smilexample{blob_measures.py}
   */
  template <class T>
  std::map<T, double> blobsArea(const Image<T> &imLbl, const bool onlyNonZero = true)
  {
    return processBlobMeasure<T, T, measAreaFunc<T>>(imLbl, onlyNonZero);
  }

  /**
   *
   * blobsArea() - Measure areas from a pre-generated Blob map (faster).
   *
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, double></b> with the @b area of each
   * blob.
   *
   */
  template <class labelT>
  std::map<labelT, double> blobsArea(std::map<labelT, Blob> &blobs)
  {
    Image<labelT> fakeImg(1, 1);
    return processBlobMeasure<labelT, labelT, measAreaFunc<labelT>>(fakeImg,
                                                                    blobs);
  }

  /**
   * blobsVolume() - Measure the sum of values of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, double></b> with the @b volume (sum of
   * pixel values) in each blob.
   */
  template <class T, class labelT>
  std::map<labelT, double> blobsVolume(const Image<T> &imIn,
                                  std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measVolFunc<T>>(imIn, blobs);
  }

  /**
   * blobsMinVal() - Measure the minimum value of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, T></b> with the @b min image value in
   * each blob.
   */
  template <class T, class labelT>
  std::map<labelT, T> blobsMinVal(const Image<T> &imIn, std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMinValFunc<T>>(imIn, blobs);
  }

  /**
   * blobsMaxVal() - Measure the maximum value of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, T></b> with the @b max image value in
   * each blob.
   */
  template <class T, class labelT>
  std::map<labelT, T> blobsMaxVal(const Image<T> &imIn, std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMaxValFunc<T>>(imIn, blobs);
  }

  /**
   * blobsRangeVal() - Measure the min and max values of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, <T, T>></b> with a vector with the @b min
   * and @b max pixel values in each blob.
   */
  template <class T, class labelT>
  std::map<labelT, std::vector<T>> blobsRangeVal(const Image<T> &imIn,
                                       std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMinMaxValFunc<T>>(imIn, blobs);
  }

  /**
   * blobsMeanVal() - Measure the mean value and the std dev. of each blob in
   * imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, <double, double>></b> with a vector with
   * the @b mean and <b> standard deviation</b> pixel values in each blob.
   */
  template <class T, class labelT>
  std::map<labelT, Vector_double> blobsMeanVal(const Image<T> &imIn,
                                          std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMeanValFunc<T>>(imIn, blobs);
  }

  /**
   * blobsValueList() - Measure the list of values of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, vector<T>></b> with the list of pixel
   * values in each blob.
   */
  template <class T, class labelT>
  std::map<labelT, std::vector<T>> blobsValueList(const Image<T> &imIn,
                                    std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, valueListFunc<T>>(imIn, blobs);
  }

  /**
   * blobsModeVal() - Measure the mode value of imIn in each blob.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, T></b> with the @b mode value in each
   * blob.
   */
  template <class T, class labelT>
  std::map<labelT, T> blobsModeVal(const Image<T> &imIn, std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measModeValFunc<T>>(imIn, blobs);
  }
  /**
   * blobsMedianVal() - Measure the median value of imIn in each blob.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, T></b> with the @b median value in each
   * blob.
   */
  template <class T, class labelT>
  std::map<labelT, T> blobsMedianVal(const Image<T> &imIn, std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measMedianValFunc<T>>(imIn, blobs);
  }

  /**
   * blobsBarycenter() - Measure barycenter of a labeled image.
   *
   * @param[in] imLbl : input labeled image
   * @param[in] onlyNonZero : skip a blob having a null label
   * @return a map of pairs <b><label, Vector<double>></b> with the coordinates
   * of the @b barycenter of each blob.
   *
   * @smilexample{blob_measures.py}
   */
  template <class T>
  std::map<T, Vector_double> blobsBarycenter(const Image<T> &imLbl,
                                        const bool onlyNonZero = true)
  {
    return processBlobMeasure<T, T, measBarycenterFunc<T>>(imLbl, onlyNonZero);
  }

  /**
   * blobsBarycenter() - Measure the barycenter of each blob in imIn.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, Vector<double>></b> with the coordinates
   * of the @b barycenter of each blob.
   */
  template <class T, class labelT>
  std::map<labelT, Vector_double> blobsBarycenter(const Image<T> &imIn,
                                             std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measBarycenterFunc<T>>(imIn, blobs);
  }

  /**
   * blobsBoundBox() - Measure bounding boxes of labeled image.
   *
   * @param[in] imLbl : input labeled image
   * @param[in] onlyNonZero : skip a blob having a null label
   * @return a map of pairs <b><label, Vector<size_t>></b> with the coordinates
   * of the <b>bounding box</b> of each label.
   */
  template <class T>
  std::map<T, std::vector<size_t>> blobsBoundBox(const Image<T> &imLbl,
                                        const bool onlyNonZero = true)
  {
    return processBlobMeasure<T, T, measBoundBoxFunc<T>>(imLbl, onlyNonZero);
  }

  /**
   * blobsBoundBox() - Measure bounding boxes of each blob (faster with blobs
   * pre-generated).
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @return a map of pairs <b><label, Vector<size_t>></b> with the coordinates
   * of the <b>bounding box</b> of each label.
   */
  template <class T, class labelT>
  std::map<labelT, std::vector<size_t>> blobsBoundBox(const Image<T> &imIn,
                                             std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measBoundBoxFunc<T>>(imIn, blobs);
  }

  /**
   * blobsMoments() - Measure blobs image moments (faster with blobs
   * pre-generated).
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   * @param[in] central : moments are evaluated with respect to the blob
   * barycenter
   *
   * @return a map of pairs <b><label, Vector<double>></b> with the first and
   * second order <b>image moments</b> of each label.
   *
   * @see measMoments()
   *
   * @smilexample{Calculating moments, example-blob-moments.py}
   *
   * @smilexample{Getting blob orientation in space, inertia_moments.py}
   */
  template <class T, class labelT>
  std::map<labelT, Vector_double> blobsMoments(const Image<T> &imIn,
                                             std::map<labelT, Blob> &blobs,
                                             bool central = false)
  {
    std::map<labelT, Vector_double> bmoments;
    bmoments = processBlobMeasure<T, labelT, measMomentsFunc<T>>(imIn, blobs);
    if (central) {
      typedef typename std::map<labelT, Vector_double>::iterator mIter;
      for (mIter it = bmoments.begin(); it != bmoments.end(); it++) {
        it->second = centerMoments(it->second);
      }
    }
    return bmoments;
  }

  /**
   * blobsMoments() - Measure image moments of each label.
   *
   * @param[in] imLbl : input @b labeled image
   * @param[in] onlyNonZero : skip a blob having a null label

   * @param[in] central : moments are evaluated with respect to the blob
   * barycenter
   *
   * @return a map of pairs <b><label, Vector<double>></b> with the <b>image
   * moments</b> of each label.
   *
   * @note
   * Whenever possible it's better to privilegiate the other version of this
   * function, as you have more control on what you get.
   * @see measMoments()
   */
  template <class T>
  std::map<T, Vector_double> blobsMoments(const Image<T> &imLbl,
                                        const bool onlyNonZero = true,
                                        bool central          = false)
  {
    std::map<T, Vector_double> bmoments;
    bmoments = processBlobMeasure<T, T, measMomentsFunc<T>>(imLbl, onlyNonZero);
    if (central) {
      typedef typename std::map<T, Vector_double>::iterator mIter;
      for (mIter it = bmoments.begin(); it != bmoments.end(); it++) {
        it->second = centerMoments(it->second);
      }
    }
    return bmoments;
  }

  /**
   * blobsEntropy() - Measure blobs entropy.
   *
   * @param[in] imIn : input labeled image
   * @param[in] blobs : input Blob map
   *
   * @return a map of pairs <b><label, double></b> with the entropy of
   * the image in each blob.
   *
   * @see measEntropy()
   */
  template <class T, class labelT>
  std::map<labelT, double> blobsEntropy(const Image<T> &imIn,
                                       std::map<labelT, Blob> &blobs)
  {
    return processBlobMeasure<T, labelT, measEntropyFunc<T>>(imIn, blobs);
  }

  /** @}*/

} // namespace smil

#endif // _D_BLOB_MEASURES_HPP
