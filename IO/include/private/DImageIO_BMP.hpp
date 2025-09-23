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

#ifndef _D_IMAGE_IO_BMP_H
#define _D_IMAGE_IO_BMP_H

#include <fstream>
#include <iostream>

#include "IO/include/private/DImageIO.hpp"
#include "Core/include/private/DImage.hpp"

namespace smil
{
  /**
   * @addtogroup IO
   */
  /* *@{*/

  struct BMPHeader;

  RES_T getBMPFileInfo(const char *filename, ImageFileInfo &fInfo);

  template <class T>
  class Image;

  template <class T = void>
  class BMPImageFileHandler : public ImageFileHandler<T>
  {
  public:
    BMPImageFileHandler() : ImageFileHandler<T>("BMP")
    {
    }

    virtual RES_T getFileInfo(const char *filename, ImageFileInfo &fInfo)
    {
      fInfo.filename = filename;
      return getBMPFileInfo(filename, fInfo);
    }

    virtual RES_T read(const char *filename, Image<T> &image)
    {
      return ImageFileHandler<T>::read(filename, image);
    }
    virtual RES_T write(const Image<T> &image, const char *filename)
    {
      return ImageFileHandler<T>::write(image, filename);
    }
  };

  // Specializations
  IMAGEFILEHANDLER_TEMP_SPEC(BMP, UINT8);
#ifdef SMIL_WRAP_RGB
  IMAGEFILEHANDLER_TEMP_SPEC(BMP, RGB);
#endif // SMIL_WRAP_RGB

  /* *@}*/

} // namespace smil

#endif // _D_IMAGE_IO_BMP_H
