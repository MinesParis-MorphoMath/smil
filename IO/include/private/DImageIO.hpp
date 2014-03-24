/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _D_IMAGE_IO_HPP
#define _D_IMAGE_IO_HPP




#include "IO/include/DCommonIO.h"
#include "Base/include/private/DImageArith.hpp"



namespace smil
{
  
    /** 
    * \addtogroup IO
    */
    /*@{*/
    

    template <class T=void>
    class ImageFileHandler
    {
    public:
      ImageFileHandler(const char *ext)
	: fileExtention(ext)
      {
      }
      const char *fileExtention;
      
      virtual RES_T getFileInfo(const char* filename, ImageFileInfo &fInfo) 
      { 
	  return RES_ERR; 
      }
      
      virtual RES_T read(const char* filename)
      {
	  return RES_ERR;
      }
      
      virtual RES_T read(const char* filename, Image<T> &image)
      {
	  cout << "Data type not implemented for " << fileExtention << " files." << endl;
	  return RES_ERR;
      }
      virtual RES_T write(const Image<T> &image, const char* filename)
      {
	  cout << "Data type not implemented for " << fileExtention << " files." << endl;
	  return RES_ERR;
      }
    };
     
    template <class T>
    ImageFileHandler<T> *getHandlerForFile(const char* filename);
    
    /**
    * Read image file
    */
    template <class T>
    RES_T read(const char* filename, Image<T> &image);

    /**
    * Read a stack of 2D images
    * 
    * The output 3D image will have the width and height of the first (2D) image and the number of images for depth.
    */
    template <class T>
    RES_T read(const vector<string> fileList, Image<T> &image);

    /**
    * Write image file
    */
    template <class T>
    RES_T write(Image<T> &image, const char *filename);
    
    RES_T getFileInfo(const char *filename, ImageFileInfo &fInfo);
    
    BaseImage *createFromFile(const char *filename);

/*@}*/

} // namespace smil



#endif // _D_IMAGE_IO_HPP
