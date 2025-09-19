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

#ifndef _DMORPHO_DISTANCE_H
#define _DMORPHO_DISTANCE_H

#include "Core/include/DCore.h"
#include "Core/include/DImage.h"
#include "DMorphoInstance.h"
#include "DStructuringElement.h"

namespace smil
{
  /**
   * @ingroup Distance
   * @{
   */

  /**
   * @brief distance() - Distance function
   * @param[in] imIn : Binary input image
   * @param[out] imOut : Output image
   * @param[in] se : Structuring Element
   */
  template <class T1, class T2>
  RES_T distance(const Image<T1> &imIn, Image<T2> &imOut,
                 const StrElt &se = DEFAULT_SE);

  /** @cond */
  /**
   * @brief distanceEuclidean() - Euclidean distance function.
   * @param[in] imIn : Binary input image
   * @param[out] imOut : Output image
   */
  template <class T1, class T2>
  RES_T distanceEuclideanOld(const Image<T1> &imIn, Image<T2> &imOut);
  /** @endcond */

  /**
   * @brief distanceEuclidean() - Euclidean distance function.
   * @param[in] imIn : Binary input image
   * @param[out] imOut : Output image
   * @param[in] se : Structuring Element
   */
  template <class T1, class T2>
  RES_T distanceEuclidean(const Image<T1> &imIn, Image<T2> &imOut,
                              const StrElt &se = DEFAULT_SE);

  /** @cond */
  template <class T1, class T2>
  RES_T dist_euclidean(const Image<T1> &imIn, Image<T2> &imOut);
  /** @endcond */

  /**
   * @brief distanceGeodesic() - Geodesic distance function
   * @param[in] imIn : Binary input image
   * @param[in] imMask : Binary mask image
   * @param[out] imOut : Output image
   * @param[in] se : Structuring Element
   *
   * @smilexample{example-distance-geodesic.py}
   */
  template <class T1, class T2>
  RES_T distanceGeodesic(const Image<T1> &imIn, const Image<T1> &imMask,
                         Image<T2> &imOut, const StrElt &se = DEFAULT_SE);

  /*
   * Internal functions - old deprecated prototypes - not to be called by user
   * programs
   */
  /** @cond */
  template <class T1, class T2>
  RES_T dist(const Image<T1> &imIn, Image<T2> &imOut,
             const StrElt &se = DEFAULT_SE)
  {
    return distance(imIn, imOut, se);
  }

  /**
   * @brief distGeneric() - Generic Distance function.
   * @param imIn : Input Image
   * @param imOut : Output Image
   * @param se : Structuring Element
   */
  template <class T1, class T2>
  RES_T distGeneric(const Image<T1> &imIn, Image<T2> &imOut,
                    const StrElt &se = DEFAULT_SE);

  /**
   * @brief distCross3d() - Distance function using a Cross3DSE Structuring
   * Element
   * @param imIn : Input Image
   * @param imOut : Output Image
   */
  template <class T1, class T2>
  RES_T distCross3d(const Image<T1> &imIn, Image<T2> &imOut);

  /**
   * @brief distCross() - Distance  function using a CrossSE Structuring Element
   * @param imIn : Input Image
   * @param imOut : Output Image
   */
  template <class T1, class T2>
  RES_T distCross(const Image<T1> &imIn, Image<T2> &imOut);

  /**
   * @brief distSquare() - Distance  function using a Square Structuring Element
   * @param imIn : Input Image
   * @param imOut : Output Image
   */
  template <class T1, class T2>
  RES_T distSquare(const Image<T1> &imIn, Image<T2> &imOut);

  template <class T1, class T2>
  RES_T distEuclidean(const Image<T1> &imIn, Image<T2> &imOut)
  {
    return distanceEuclidean(imIn, imOut);
  }

  template <class T1, class T2>
  RES_T distGeodesic(const Image<T1> &imIn, const Image<T1> &imMask,
                     Image<T2> &imOut, const StrElt &se = DEFAULT_SE)
  {
    return distanceGeodesic(imIn, imMask, imOut, se);
  }
  /** @endcond */

  /** @cond */
  /**
   * @brief Distance function performed with successive erosions.
   * @param imIn : Input Image
   * @param imOut : Output Image
   * @param se : Structuring Element
   */
  template <class T>
  RES_T distV0(const Image<T> &imIn, Image<T> &imOut,
               const StrElt &se = DEFAULT_SE);
  /** @endcond */

  /** @}*/
} // namespace smil

#include "private/DMorphoDistance.hpp"

#include "private/DMorphoEuclidean.hpp"

#endif // _DMORPHO_DISTANCE_H
