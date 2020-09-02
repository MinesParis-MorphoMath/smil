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

#ifndef _D_MORPHO_FILTER_HPP
#define _D_MORPHO_FILTER_HPP

#include "Core/include/DImage.h"
#include "DMorphImageOperations.hxx"

#include "Base/include/private/DImageArith.hpp"
#include "Morpho/include/DMorphoInstance.h"
#include "DHitOrMiss.hpp"

namespace smil
{
  /**
   * @ingroup Morpho
   * @defgroup Filters Morphological Filters
   * @{
   */

  /**
   * close() - Morphological grayscale closing
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T close(const Image<T> &imIn, Image<T> &imOut,
              const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);
    ImageFreezer freeze(imOut);

    bool inplaceSafe = MorphImageFunction<T, supLine<T>>::isInplaceSafe(se);
    Image<T> *imTmp;
    if (inplaceSafe)
      imTmp = &imOut;
    else
      imTmp = new Image<T>(imIn);

    ASSERT((dilate(imIn, *imTmp, se) == RES_OK));
    ASSERT((erode(*imTmp, imOut, se) == RES_OK));

    if (!inplaceSafe)
      delete imTmp;

    return RES_OK;
  }

  /**
   * close() - Morphological grayscale closing using the default structuring
   * element but being able to set its size
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] seSize : size of structuring element
   *
   * @note
   * If you want to use a structuring element different of the default
   * you should set it before
   */
  template <class T>
  RES_T close(const Image<T> &imIn, Image<T> &imOut, UINT seSize)
  {
    return close(imIn, imOut, DEFAULT_SE(seSize));
  }

  /**
   * open() - Morphological grayscale opening
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T open(const Image<T> &imIn, Image<T> &imOut,
             const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);
    ImageFreezer freeze(imOut);

    bool inplaceSafe = MorphImageFunction<T, supLine<T>>::isInplaceSafe(se);
    Image<T> *imTmp;
    if (inplaceSafe)
      imTmp = &imOut;
    else
      imTmp = new Image<T>(imIn);

    ASSERT((erode(imIn, *imTmp, se) == RES_OK));
    ASSERT((dilate(*imTmp, imOut, se) == RES_OK));

    if (!inplaceSafe)
      delete imTmp;

    return RES_OK;
  }

  /**
   * open() - Morphological grayscale opening using the default structuring
   * element but being able to set its size
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] seSize : size of structuring element
   *
   * @note
   * If you want to use a structuring element different of the default
   * you should set it before
   */
  template <class T>
  RES_T open(const Image<T> &imIn, Image<T> &imOut, UINT seSize)
  {
    return open(imIn, imOut, DEFAULT_SE(seSize));
  }


  /**
   * Alternate Sequential Filter beginning by a closing
   *
   * Sequence of closings and openings with increasing size <b>1, 2, ...,
   * max_size</b>. The @b max_size is given by the size of the structuring
   * element (for example @b 3 for @b hSE(3)).
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element with the maximum size of the filter
   *
   * @smilexample{example-asfclose.py}
   */
  template <class T>
  RES_T asfClose(const Image<T> &imIn, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn, true); // clone
    for (UINT i = 1; i <= se.size; i++) {
      ASSERT((close(tmpIm, imOut, se(i)) == RES_OK));
      ASSERT((open(imOut, tmpIm, se(i)) == RES_OK));
    }
    ASSERT((copy(tmpIm, imOut) == RES_OK));

    return RES_OK;
  }

  /**
   * Alternate Sequential Filter beginning by an opening
   *
   * Sequence of openings and closings with increasing size <b>1, 2, ...,
   * max_size</b>. The @b max_size is given by the size of the structuring
   * element (for example @b 3 for @b hSE(3)).
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element with the maximum size of the filter
   */
  template <class T>
  RES_T asfOpen(const Image<T> &imIn, Image<T> &imOut,
                const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn, true); // clone
    for (UINT i = 1; i <= se.size; i++) {
      ASSERT((open(tmpIm, imOut, se(i)) == RES_OK));
      ASSERT((close(imOut, tmpIm, se(i)) == RES_OK));
    }
    ASSERT((copy(tmpIm, imOut) == RES_OK));

    return RES_OK;
  }

  /** @cond */
  template <class T> class meanFunct : public MorphImageFunctionBase<T, T>
  {
  public:
    typedef MorphImageFunctionBase<T, T> parentClass;

    virtual inline void processPixel(size_t pointOffset,
                                     vector<int> &dOffsetList)
    {
      double meanVal                = 0;
      vector<int>::iterator dOffset = dOffsetList.begin();
      while (dOffset != dOffsetList.end()) {
        meanVal += double(parentClass::pixelsIn[pointOffset + *dOffset]);
        dOffset++;
      }
      parentClass::pixelsOut[pointOffset] =
          T(meanVal / double(dOffsetList.size()));
    }
  };
  /** @endcond */

  /**
   * Mean filter
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T mean(const Image<T> &imIn, Image<T> &imOut,
             const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    meanFunct<T> f;

    ASSERT((f._exec(imIn, imOut, se) == RES_OK));

    return RES_OK;
  }

  /** @cond */
  template <class T> class medianFunct : public MorphImageFunctionBase<T, T>
  {
  public:
    typedef MorphImageFunctionBase<T, T> parentClass;

    virtual inline void processPixel(size_t pointOffset,
                                     vector<int> &dOffsetList)
    {
      vector<T> vals;
      vector<int>::iterator dOffset = dOffsetList.begin();
      while (dOffset != dOffsetList.end()) {
        vals.push_back(parentClass::pixelsIn[pointOffset + *dOffset]);
        dOffset++;
      }
      sort(vals.begin(), vals.end());
      parentClass::pixelsOut[pointOffset] = vals[dOffsetList.size() / 2];
    }
  };
  /** @endcond */

  /**
   * Median filter
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T median(const Image<T> &imIn, Image<T> &imOut,
               const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    medianFunct<T> f;

    ASSERT((f._exec(imIn, imOut, se) == RES_OK));

    return RES_OK;
  }

  /** @cond */
  template <class T> class rankFunct : public MorphImageFunctionBase<T, T>
  {
  public:
    typedef MorphImageFunctionBase<T, T> parentClass;

    rankFunct(double per) : MorphImageFunctionBase<T, T>(), percentile(per)
    {
    }
    virtual inline void processPixel(size_t pointOffset,
                                     vector<int> &dOffsetList)
    {
      vector<T> vals;
      vector<int>::iterator dOffset = dOffsetList.begin();
      while (dOffset != dOffsetList.end()) {
        vals.push_back(parentClass::pixelsIn[pointOffset + *dOffset]);
        dOffset++;
      }
      sort(vals.begin(), vals.end());
      parentClass::pixelsOut[pointOffset] =
          vals[static_cast<int>(dOffsetList.size() * this->percentile)];
    }

  private:
    double percentile;
  };
  /** @endcond */

  /**
   * Rank filter
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] percentile :
   * @param[in] se : structuring element
   */
  template <class T>
  RES_T rank(const Image<T> &imIn, Image<T> &imOut, double percentile,
             const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    rankFunct<T> f(percentile);

    ASSERT((f._exec(imIn, imOut, se) == RES_OK));

    return RES_OK;
  }

  /** @}*/

} // namespace smil

#endif // _D_MORPHO_FILTER_HPP
