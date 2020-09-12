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

#ifndef _D_BLOB_OPERATIONS_HPP
#define _D_BLOB_OPERATIONS_HPP

#include "Core/include/private/DImage.hpp"
#include "DImageHistogram.hpp"
#include "DMeasures.hpp"
#include "DBlobMeasures.hpp"
#include "Morpho/include/DMorpho.h"

#include <map>

using namespace std;

namespace smil
{
  /**
   * @addtogroup BlobOps
   * @{
   */

  /** @cond */
  class BlobOps
  {
    BlobOps(){};

    ~BlobOps(){};
  };
  /** @endcond */

  /** areaThreshold() -
   *
   * @param[in] imIn : input image (@b binary or @b labeled)
   * @param[in] threshold : threshold level
   * @param[in] gt : blobs which area is @txtbold{greater than} will be retained
   *   if @b gt is @b true, and @txtbold{lesser than} if @b gt is @b false.
   * @param[out] imOut : output image
   *
   * @note
   * - output image is of the same kind of input image : @b binary or
   *   @b labeled.
   *
   * @smilexample{example-areathreshold.py}
   */
  template <typename T1, typename T2>
  RES_T areaThreshold(const Image<T1> &imIn, const int threshold,
                           const bool gt, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    map<T2, Blob> blobs;
    Image<T2> imLabel(imOut);

    bool imBinary = false;
    if (isBinary(imIn)) {
      imBinary = true;
      label(imIn, imLabel);
    } else
      copy(imIn, imLabel);
    blobs = computeBlobs(imLabel, true);

    typedef typename map<T2, Blob>::iterator blobIter;
    typedef typename map<T2, double>::iterator areaIter;

    map<T2, double> areas = measAreas(blobs);

    map<T2, T2> lut;
    areaIter ait;
    for (ait = areas.begin(); ait != areas.end(); ait++) {
      if ((gt && ait->second < threshold) || (!gt && ait->second > threshold)) {
        blobIter bit = blobs.find(ait->first);
        if (bit != blobs.end())
          blobs.erase(bit);
        lut[ait->first] = 0;
      } else {
        if (imBinary)
          lut[ait->first] = ImDtTypes<T2>::max();
        else
          lut[ait->first] = ait->first;
      }
    }

    fill<T2>(imOut, T2(0));
    ImageFreezer freeze(imOut);

    T2 *pl          = imLabel.getPixels();
    T2 *po          = imOut.getPixels();
    size_t nbPixels = imOut.getPixelCount();
#ifdef USE_OPEN_MP
    int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif
    for (size_t i = 0; i < nbPixels; i++)
      po[i] = lut[pl[i]];

    return RES_OK;
  }

  /*
   * #    #    #  ######  #####    #####     #      ##
   * #    ##   #  #       #    #     #       #     #  #
   * #    # #  #  #####   #    #     #       #    #    #
   * #    #  # #  #       #####      #       #    ######
   * #    #   ##  #       #   #      #       #    #    #
   * #    #    #  ######  #    #     #       #    #    #
   */

  /** inertiaMatrices() -
   *
   * @param[in] imIn : input image
   * @param[in] blobs : blobs in the image
   * @param[in] central : blobs centered on their barycenters
   *
   * @smilexample{example-inertia-matrix.py}
   */
  template <typename T, typename labelT>
  map<labelT, Vector_double> inertiaMatrices(const Image<T> &imIn,
                                        map<labelT, Blob> &blobs,
                                        const bool central = false)
  {
    map<labelT, Vector_double> inertia;

    if (!imIn.isAllocated())
    {
      ERR_MSG("Input image not allocated !!!");
      return inertia;
    }

    bool im3d = (imIn.getDimension() == 3);

    map<labelT, Vector_double> moments = measBlobMoments(imIn, blobs, central);

    typedef typename map<labelT, Blob>::iterator blobIter;
    for (blobIter it = blobs.begin(); it != blobs.end(); it++) {
      Vector_double m(3);

      if (im3d) {
        Vector_double mr(9, 0.);

        if (moments[it->first].size() != 10)
        {
          ERR_MSG("Not enough moments...");
          continue;
        }
        mr[0] = moments[it->first][8] + moments[it->first][9];
        mr[1] = - moments[it->first][4];
        mr[2] = - moments[it->first][5];

        mr[3] = - moments[it->first][4];
        mr[4] = moments[it->first][7] + moments[it->first][9];
        mr[5] = - moments[it->first][6];

        mr[6] = - moments[it->first][5];
        mr[7] = - moments[it->first][6];
        mr[8] = moments[it->first][7] + moments[it->first][8];
        m = mr;
        inertia[it->first] = mr;
      } else {
        Vector_double mr(4, 0.);

        if (moments[it->first].size() != 6)
        {
          ERR_MSG("Not enough moments...");
          continue;
        }
        mr.resize(4);
        mr[0] = moments[it->first][4];
        mr[1] = - moments[it->first][3];
        
        mr[2] = - moments[it->first][3];
        mr[3] = moments[it->first][5];
        m = mr;
        inertia[it->first] = mr;
      }
    }
    return inertia;
  }

  /** inertiaMatrices() -
   *
   * @param[in] imLbl : input @b labeled image
   * @param[in] onlyNonZero :
   * @param[in] central : blobs centered on their barycenters
   */
  template <typename T>
  map<T, Vector_double> inertiaMatrices(const Image<T> &imLbl,
                                        const bool onlyNonZero = true,
                                        const bool central    = false)
  {
    map<T, Vector_double> inertia;

    if (!imLbl.isAllocated())
    {
      ERR_MSG("Input image not allocated !!!");
      return inertia;
    }

    map<T, Blob> blobs            = computeBlobs(imLbl, onlyNonZero);
#if 1
    return inertiaMatrices(imLbl, blobs, central);
#else
    map<T, Vector_double> moments = measBlobMoments(imLbl, blobs);

    bool im3d = (imLbl.getDimension() == 3);
 
    typedef typename map<T, Blob>::iterator blobIter;
    for (blobIter it = blobs.begin(); it != blobs.end(); it++) {
      if (central)
        moments[it->first] = centerMoments(moments[it->first]);

      Vector_double m(3);
      if (im3d) {
        Vector_double mr(9, 0.);

        if (moments[it->first].size() != 10)
        {
          ERR_MSG("Not enough moments...");
          continue;
        }
        mr[0] = moments[it->first][8] + moments[it->first][9];
        mr[1] = - moments[it->first][4];
        mr[2] = - moments[it->first][5];

        mr[3] = - moments[it->first][4];
        mr[4] = moments[it->first][7] + moments[it->first][9];
        mr[5] = - moments[it->first][6];

        mr[6] = - moments[it->first][5];
        mr[7] = - moments[it->first][6];
        mr[8] = moments[it->first][7] + moments[it->first][8];
        m = mr;
        inertia[it->first] = mr;
      } else {
        Vector_double mr(4, 0.);

        if (moments[it->first].size() != 6)
        {
          ERR_MSG("Not enough moments...");
          continue;
        }
        mr.resize(4);
        mr[0] = moments[it->first][4];
        mr[1] = - moments[it->first][3];
        
        mr[2] = - moments[it->first][3];
        mr[3] = moments[it->first][5];
        m = mr;
        inertia[it->first] = mr;
      }
    }
    return inertia;
#endif
  }

  /** @}*/

} // namespace smil

#endif // _D_BLOB_OPERATIONS_HPP
