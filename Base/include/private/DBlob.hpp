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

#ifndef _D_BLOB_HPP
#define _D_BLOB_HPP

#include "Core/include/private/DImage.hpp"
#include <map>

namespace smil
{
  /**
   * @ingroup Base
   * @defgroup BlobMesures Mesures on blobs
   * @{
   * @b Blobs
   *
   * Blobs are just maps (or dictionaries in Python) to represent disjoint
   * regions in image.
   *
   * @b blobs are probably the best way to handle measures on regions of images,
   * as calling a function to evaluate a measure on a @b blobs map is equivalent
   * to call the equivalent function for each @b blob. E.g. calling blobsArea()
   * is equivalent to call area() for each @b blob in the map.
   *
   * To handle blobs, you need to have a <b>labeled image</b>, generated by some
   * method of segmentation (e.g. watershed()), and create them thanks to
   * computeBlobs().
   *
   * @smilexample{example-intro-blobs.py}
   *
   */

  /**
   * Start offset and length of pixels in an image
   */
  struct PixelSequence {
    size_t offset;
    size_t size;
    PixelSequence() : offset(0), size(0)
    {
    }
    ~PixelSequence()
    {
    }
    PixelSequence(size_t off, size_t siz) : offset(off), size(siz)
    {
    }
  };

  /**
   * List of offset and size of line contiguous pixels.
   *
   * A Blob contains a vector of PixelSequence.
   */
  struct Blob {
    vector<PixelSequence> sequences;
    typedef vector<PixelSequence>::iterator sequences_iterator;
    typedef vector<PixelSequence>::const_iterator sequences_const_iterator;
    typedef vector<PixelSequence>::const_reverse_iterator
        sequences_const_reverse_iterator;
  };

  /**
   * Create a map of blobs from a labeled image
   *
   * @param[in] imIn : input @b labeled image
   * @param[in] onlyNonZero : ignore regions whose label is @b 0
   * @return a map of pairs <b><label, blob></b>
   */
  template <class T>
  map<T, Blob> computeBlobs(const Image<T> &imIn, bool onlyNonZero = true)
  {
    map<T, Blob> blobs;

    ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, blobs);

    typename ImDtTypes<T>::sliceType lines = imIn.getLines();
    typename ImDtTypes<T>::lineType pixels;
    size_t npix   = imIn.getWidth();
    size_t nlines = imIn.getLineCount();

    T curVal;

    for (size_t l = 0; l < nlines; l++) {
      size_t curSize  = 0;
      size_t curStart = l * npix;

      pixels = lines[l];
      curVal = pixels[0];
      if (curVal != 0 || !onlyNonZero)
        curSize++;

      for (size_t i = 1; i < npix; i++) {
        if (pixels[i] == curVal)
          curSize++;
        else {
          if (curVal != 0 || !onlyNonZero)
            blobs[curVal].sequences.push_back(PixelSequence(curStart, curSize));
          curStart = i + l * npix;
          curSize  = 1;
          curVal   = pixels[i];
        }
      }
      if (curVal != 0 || !onlyNonZero)
        blobs[curVal].sequences.push_back(PixelSequence(curStart, curSize));
    }

    return blobs;
  }

  /**
   * Represent Blobs in an image
   *
   * @param[in] blobs : input blobs map
   * @param[in] imOut : output image
   * @param[in] blobsValue : value to assign to each blob
   * @param[in] fillFirst : fill the background before beginning
   * @param[in] defaultValue : default value for the background (default : 0)
   *
   * @note
   * If <c>blobsValue == 0</c>, each blobs is represented with its label value.
   */
  template <class labelT, class T>
  RES_T drawBlobs(map<labelT, Blob> &blobs, Image<T> &imOut,
                  T blobsValue = ImDtTypes<T>::max(), bool fillFirst = true,
                  T defaultValue = T(0))
  {
    ASSERT_ALLOCATED(&imOut);

    ImageFreezer freeze(imOut);

    if (fillFirst)
      ASSERT(fill(imOut, defaultValue) == RES_OK);

    typename ImDtTypes<T>::lineType pixels = imOut.getPixels();
    size_t pixCount                        = imOut.getPixelCount();
    bool allBlobsFit                       = true;

    typename map<labelT, Blob>::const_iterator blob_it;
    for (blob_it = blobs.begin(); blob_it != blobs.end(); blob_it++) {
      // Verify that the blob can fit in the image
      Blob::sequences_const_reverse_iterator last =
          blob_it->second.sequences.rbegin();
      if ((*last).offset + (*last).size > pixCount) {
        allBlobsFit = false;
      } else {
        Blob::sequences_const_iterator it = blob_it->second.sequences.begin();
        Blob::sequences_const_iterator it_end = blob_it->second.sequences.end();

        T outVal = blobsValue != defaultValue ? blobsValue : blob_it->first;
        for (; it != it_end; it++) {
          typename ImDtTypes<T>::lineType line = pixels + (*it).offset;
          for (size_t i = 0; i < it->size; i++)
            line[i] = outVal;
        }
      }
    }
    imOut.modified();

    ASSERT(allBlobsFit, "Some blobs are outside the image", RES_ERR);

    return RES_OK;
  }

  /**
   * Represent Blobs in an image with a lookup map.
   *
   * @param[in] blobs : input blobs map
   * @param[in] lut : lookup table with value to assign to each blob
   * @param[in] imOut : output image
   * @param[in] fillFirst : fill the background before beginning
   * @param[in] defaultValue : default value for the background (default : 0)
   *
   */
  template <class labelT, class T>
  RES_T drawBlobs(map<labelT, Blob> &blobs, map<labelT, T> &lut,
                  Image<T> &imOut, bool fillFirst = true, T defaultValue = T(0))
  {
    ASSERT_ALLOCATED(&imOut);

    ImageFreezer freeze(imOut);

    if (fillFirst)
      ASSERT(fill(imOut, defaultValue) == RES_OK);

    typename ImDtTypes<T>::lineType pixels = imOut.getPixels();
    size_t pixCount                        = imOut.getPixelCount();
    bool allBlobsFit                       = true;

    typename map<labelT, Blob>::const_iterator blob_it;
    for (blob_it = blobs.begin(); blob_it != blobs.end(); blob_it++) {
      // Verify that the blob can fit in the image
      Blob::sequences_const_reverse_iterator last =
          blob_it->second.sequences.rbegin();
      if ((*last).offset + (*last).size > pixCount) {
        allBlobsFit = false;
      } else {
        Blob::sequences_const_iterator it = blob_it->second.sequences.begin();
        Blob::sequences_const_iterator it_end = blob_it->second.sequences.end();

        typename map<labelT, T>::const_iterator valIt =
            lut.find(blob_it->first);
        T outVal = valIt != lut.end() ? valIt->second : defaultValue;
        for (; it != it_end; it++) {
          typename ImDtTypes<T>::lineType line = pixels + (*it).offset;
          for (size_t i = 0; i < it->size; i++)
            line[i] = outVal;
        }
      }
    }
    imOut.modified();

    ASSERT(allBlobsFit, "Some blobs are outside the image", RES_ERR);

    return RES_OK;
  }

  /**
   * getBlobIDFromOffset() - get the blob ID which contains a pixel (or voxel)
   * given its offset
   *
   * @param[in] blobs  : the @b blobs map
   * @param[in] offset : the offset to look for
   * @returns the ID of the blob containing the offset, or @TB{0} if the offset
   * is in the background.
   *
   */
  template <class labelT>
  int getBlobIDFromOffset(map<labelT, Blob> &blobs, size_t offset)
  {
    int id = 0;

    typename map<labelT, Blob>::const_iterator blob_it;
    typedef typename Blob::sequences_const_iterator seqit_t;

    for (blob_it = blobs.begin(); blob_it != blobs.end(); blob_it++) {
      seqit_t it_end = blob_it->second.sequences.end();

      for(seqit_t it = blob_it->second.sequences.begin(); it != it_end; it++)
      {
        if (offset >= it->offset && offset < it->offset + it->size)
          id = blob_it->first;
      }
      if (id > 0)
        break;
    }

    return id;
  }

  /**
   * getBlobIDFromOffset() - get the blob ID which contains a pixel (or voxel)
   * given its coordinates
   *
   * @param[in] blobs  : the @b blobs map
   * @param[in] imIn : input image where blobs are defined
   * @param[in] x, y[, z] : point coordinates
   * @returns the ID of the blob containing pixel (voxel), or @TB{0} if in the
   * background.
   *
   */
  template <class labelT, class T>
  int getBlobIDFromOffset(Image<T> imIn, map<labelT, Blob> &blobs, int x, int y, int z = 0)
  {
    int id = 0;

    size_t offset = imIn.getOffsetFromCoords(x, y, z);

    typename map<labelT, Blob>::const_iterator blob_it;
    typedef typename Blob::sequences_const_iterator seqit_t;

    for (blob_it = blobs.begin(); blob_it != blobs.end(); blob_it++) {
      seqit_t it_end = blob_it->second.sequences.end();

      for(seqit_t it = blob_it->second.sequences.begin(); it != it_end; it++)
      {
        if (offset >= it->offset && offset < it->offset + it->size)
          id = blob_it->first;
      }
      if (id > 0)
        break;
    }

    return id;
  }

  /** @} */

} // namespace smil

#endif // _D_BASE_MEASURE_OPERATIONS_HPP
