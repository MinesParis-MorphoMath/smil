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


#ifndef _D_IMAGE_IO_PNG_H
#define _D_IMAGE_IO_PNG_H


#include <iostream>

#include "Core/include/private/DTypes.hpp"
#include "Core/include/DCommon.h"
#include "Core/include/DErrors.h"

#include "NSTypes/RGB/include/DRGB.h"

using namespace std;


#ifdef USE_PNG


namespace smil
{
  
    /** 
    * \addtogroup IO
    */
    /*@{*/
    
    class PNGHeader;
    
    RES_T getPNGFileInfo(const char* filename, ImageFileInfo &fInfo);

    template <class T> class Image;

    template <class T=void>
    class PNGImageFileHandler : public ImageFileHandler<T>
    {
      public:
	PNGImageFileHandler()
	  : ImageFileHandler<T>("PNG")
	{
	}
	
	virtual RES_T getFileInfo(const char* filename, ImageFileInfo &fInfo)
	{
	    return getPNGFileInfo(filename, fInfo);
	}
	
	virtual RES_T read(const char* filename, Image<T> &image)
	{
	    return ImageFileHandler<T>::read(filename, image);
	}
	virtual RES_T write(const Image<T> &image, const char* filename)
	{
	    return ImageFileHandler<T>::write(image, filename);
	}
    };

    // Specializations
    template <>
    RES_T PNGImageFileHandler<UINT8>::read(const char *filename, Image<UINT8> &image);
    template <>
    RES_T PNGImageFileHandler<UINT16>::read(const char *filename, Image<UINT16> &image);
    template <>
    RES_T PNGImageFileHandler<RGB>::read(const char *filename, Image<RGB> &image);

    template <>
    RES_T PNGImageFileHandler<UINT8>::write(const Image<UINT8> &image, const char *filename);
    template <>
    RES_T PNGImageFileHandler<UINT16>::write(const Image<UINT16> &image, const char *filename);
    template <>
    RES_T PNGImageFileHandler<RGB>::write(const Image<RGB> &image, const char *filename);
    
/*@}*/

} // namespace smil


#endif // USE_PNG



#endif // _D_IMAGE_IO_PNG_H
