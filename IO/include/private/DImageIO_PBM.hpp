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

#ifndef _D_IMAGE_IO_PBM_H
#define _D_IMAGE_IO_PBM_H

#include <fstream>
#include <iostream>

#include "Core/include/private/DTypes.hpp"
#include "Core/include/private/DImage.hpp"
#include "IO/include/private/DImageIO.hpp"

namespace smil
{
  /**
   * @addtogroup IO
   */
  /* *@{*/

  RES_T readNetPBMFileInfo(std::ifstream &fp, ImageFileInfo &fInfo,
                           unsigned int &maxval);
  RES_T readNetPBMFileInfo(const char *filename, ImageFileInfo &fInfo,
                           unsigned int &maxval);

  template <class T> class Image;

  template <class T = void>
  class PGMImageFileHandler : public ImageFileHandler<T>
  {
  public:
    PGMImageFileHandler() : ImageFileHandler<T>("PGM")
    {
    }

    virtual RES_T getFileInfo(const char *filename, ImageFileInfo &fInfo)
    {
      unsigned int dum;
      fInfo.filename = filename;
      return readNetPBMFileInfo(filename, fInfo, dum);
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

  template <class T = void>
  class PBMImageFileHandler : public ImageFileHandler<T>
  {
  public:
    PBMImageFileHandler() : ImageFileHandler<T>("PBM")
    {
    }

    virtual RES_T getFileInfo(const char *filename, ImageFileInfo &fInfo)
    {
      unsigned int dum;
      return readNetPBMFileInfo(filename, fInfo, dum);
    }

    virtual RES_T read(const char *filename, Image<T> &image)
    {
      /* open image file */
      std::ifstream fp(filename, std::ios_base::binary);

      if (!fp.is_open()) {
        std::cout << "Cannot open file " << filename << std::endl;
        return RES_ERR_IO;
      }

      ImageFileInfo fInfo;
      unsigned int dum; // no maxval in PBM format
      ASSERT(readNetPBMFileInfo(fp, fInfo, dum) == RES_OK, RES_ERR_IO);
      ASSERT(fInfo.colorType == ImageFileInfo::COLOR_TYPE_BINARY,
             "Not an binary image", RES_ERR_IO);

      size_t width  = fInfo.width;
      size_t height = fInfo.height;

      ASSERT((image.setSize(width, height) == RES_OK), RES_ERR_BAD_ALLOCATION);

      if (fInfo.fileType == ImageFileInfo::FILE_TYPE_BINARY) {
        typename ImDtTypes<T>::sliceType lines = image.getLines();

        //  int nBytePerLine = width%8==0 ? width/8 : width/8+1;
        char val;
        int k;

        for (size_t j = 0; j < height; j++) {
          typename ImDtTypes<T>::lineType pixels = lines[j];

          for (size_t i = 0; i < width; i++) {
            if (i % 8 == 0)
              fp.read(&val, 1);

            k         = 7 - i % 8;
            pixels[i] = ((val >> k) % 2) == 0 ? T(0) : ImDtTypes<T>::max();
          }
        }
      } else {
        typename ImDtTypes<T>::lineType pixels = image.getPixels();

        int val;
        for (size_t i = 0; i < image.getPixelCount(); i++) {
          fp >> val;
          pixels[i] = val == 0 ? T(0) : ImDtTypes<T>::max();
        }
      }

      fp.close();

      return RES_OK;
    }
    virtual RES_T write(const Image<T> &image, const char *filename)
    {
      return ImageFileHandler<T>::write(image, filename);
    }
  };

  template <>
  inline RES_T PBMImageFileHandler<void>::read(const char *, Image<void> &)
  {
    return RES_ERR;
  }

  template <>
  inline RES_T PBMImageFileHandler<void>::write(const Image<void> &,
                                                const char *)
  {
    return RES_ERR;
  }

  // Specializations
  IMAGEFILEHANDLER_TEMP_SPEC(PGM, UINT8);

#ifdef SMIL_WRAP_RGB
  IMAGEFILEHANDLER_TEMP_SPEC(PGM, RGB);
#endif // SMIL_WRAP_RGB

  /* *@}*/

} // namespace smil

#endif // _D_IMAGE_IO_PBM_H
