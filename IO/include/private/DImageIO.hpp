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

#ifndef _D_IMAGE_IO_HPP
#define _D_IMAGE_IO_HPP

#include "IO/include/DCommonIO.h"
#include "Base/include/private/DImageArith.hpp"

namespace smil
{
  /**
   * @addtogroup IO
   */

  /**@{*/

  template <class T = void>
  class ImageFileHandler
  {
  public:
    ImageFileHandler(const char *ext) : fileExtention(ext)
    {
    }
    virtual ~ImageFileHandler()
    {
    }

    const char *fileExtention;

    virtual RES_T getFileInfo(const char *, ImageFileInfo &)
    {
      return RES_ERR;
    }

    virtual RES_T read(const char *)
    {
      return RES_ERR;
    }

    virtual RES_T read(const char *, Image<T> &)
    {
      T *dum = NULL;
      std::cout << getDataTypeAsString<T>(dum)
                << " data type not implemented for " << fileExtention
                << " files (read)." << std::endl;
      return RES_ERR;
    }
    virtual RES_T write(const Image<T> &, const char *)
    {
      T *dum = NULL;
      std::cout << getDataTypeAsString<T>(dum)
                << " data type not implemented for " << fileExtention
                << " files (write)." << std::endl;
      return RES_ERR;
    }
  };

  template <class T>
  ImageFileHandler<T> *getHandlerForFile(const char *filename);

  /**
   * Read image file
   *
   * @param[in] filename : filename (with path) to read
   * @param[out] image : output image
   *
   * @note
   * The following file types are recognized : @b BMP @b JPG @b PBM @b PNG @b
   * TIFF @b VTK
   *
   *
   * @smilexample{example-read.py}
   */
  template <class T>
  RES_T read(const char *filename, Image<T> &image);

  /**
   * Read a stack of 2D images and convert then into a 3D image
   *
   * @param[in] fileList : list of 2D image files to read
   * @param[out] image : output image
   *
   * @note
   * The output 3D image will have the width and height of the first (2D) image
   * and the number of images for depth.
   * @note
   * The following file types are recognized : @b BMP @b JPG @b PBM @b PNG @b
   * TIFF @b VTK
   *
   */
  template <class T>
  RES_T read(const std::vector<std::string> fileList, Image<T> &image);

  /**
   * Write image into file
   *
   * @param[in] image : image to write to file
   * @param[in] filename : file name
   */
  template <class T>
  RES_T write(const Image<T> &image, const char *filename);

  /**
   * Write a 3D image as a stack of 2D image files
   *
   * The file list must contain the same number of filenames as the 3D image
   * depth.
   *
   * @param[in] image : image to write to file
   * @param[in] fileList : list of filenames.
   *
   * @b Example:
   * @code{.py}
   *
   * im1 = Image("img3d.vtk")
   * fileNames = [ "img{:03d}.png".format(i) for i in range(im1.getDepth()) ]
   * write(im1, fileNames)
   *
   * @endcode
   */
  template <class T>
  RES_T write(const Image<T> &image, const std::vector<std::string> fileList);

  /**
   * Get information about an image file
   *
   * @param[in] filename : filename
   * @param[out] fInfo : struct ImageFileInfo with image information
   */
  RES_T getFileInfo(const char *filename, ImageFileInfo &fInfo);

  /**
   * createFromFile
   * TBD
   */
  BaseImage *createFromFile(const char *filename);

  /**@}*/

} // namespace smil

#endif // _D_IMAGE_IO_HPP
