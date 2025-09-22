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

#ifndef _D_IMAGE_IO_RAW_HPP
#define _D_IMAGE_IO_RAW_HPP

#include <cstdio>
#include <cstdlib>
#include <string>

#include "Core/include/private/DImage.hpp"

namespace smil
{
  /**
   * @addtogroup IO
   */
  /**@{*/

  /**
   * Get an image saved in a @b RAW format
   *
   * @param[in] filename : file name
   * @param[in] width, height, depth : image dimensions
   * @param[out] image : output image
   *
   * @note
   * - @b RAW format here means a format without metadata, i.e. just a stream of
   * data and not the @b RAW format from camera makers (@b NEF, @b RAF, @b CR2,
   * ...)
   *
   * @note
   * - Pixel data type and image dimensions shall be known in advance as the
   * file doesn't contains metadata
   *
   * @note
   * - At some contexts data may need to be adapted. E.g. transformed from @b
   * bigendian to @b littleendian or from @b float to @b UINT8.
   *
   * @smilexample{example-readRAW.py}
   */
  template <class T>
  RES_T readRAW(const char *filename, size_t width, size_t height, size_t depth,
                Image<T> &image)
  {
    FILE *fp = NULL;

    /* open image file */
    SMIL_OPEN(fp, filename, "rb");

    ASSERT(fp, "Error: couldn't open file", RES_ERR_IO);

    image.setSize(width, height, depth);
    //   image->allocate();

    size_t ret =
        fread(image.getVoidPointer(), sizeof(T), image.getPixelCount(), fp);
    if (ret == 0) {
      fprintf(stderr, "error reading \"%s\"!\n", filename);
      return RES_ERR;
    }

    fclose(fp);

    image.modified();

    return RES_OK;
  }

  /**
   * Save an image in a @b RAW format
   *
   * @param[in] image : input image
   * @param[in] filename : file name
   *
   * @note
   * @b RAW format here means a format without metadata, i.e. just a stream of
   * data and not the @b RAW format from camera makers (@b NEF, @b RAF, @b CR2,
   * ...)
   */
  template <class T> RES_T writeRAW(Image<T> &image, const char *filename)
  {
    FILE *fp = NULL;

    /* open image file */
    SMIL_OPEN(fp, filename, "wb");

    ASSERT(fp, "Error: couldn't open file", RES_ERR_IO);

    fwrite(image.getVoidPointer(), sizeof(T), image.getPixelCount(), fp);

    fclose(fp);

    return RES_OK;
  }

  /**@}*/

} // namespace smil

#endif // _D_IMAGE_IO_RAW_H
