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


#include "Core/include/private/DImage.hxx"

#include "IO/include/private/DImageIO.hpp"
#include "IO/include/private/DImageIO.hxx"

#ifdef SMIL_WRAP_RGB
#include "NSTypes/RGB/include/DRGB.h"
#endif // SMIL_WRAP_RGB

#include "Core/include/DErrors.h"

#include <string>
#include <algorithm>
#include <ctype.h>

#ifdef USE_CURL
#include <curl/curl.h>
#endif // USE_CURL

namespace smil
{


    RES_T getFileInfo(const char *filename, ImageFileInfo &fInfo)
    {
          auto_ptr< ImageFileHandler<void> > fHandler(getHandlerForFile<void>(filename));
          
          if (fHandler.get())
            return fHandler->getFileInfo(filename, fInfo);
          else return RES_ERR;
    }
    
    BaseImage *createFromFile(const char *filename)
    {
          string fileExt = getFileExtension(filename);
          string filePrefix = (string(filename).substr(0, 7));
          
          if (filePrefix=="http://")
          {
              BaseImage *img = NULL;
      #ifdef USE_CURL
              string tmpFileName = "_smilTmpIO." + fileExt;
              if (getHttpFile(filename, tmpFileName.c_str())!=RES_OK)
              {
                  ERR_MSG(string("Error downloading file ") + filename);
                  return img;
              }
              img = createFromFile(tmpFileName.c_str());
              remove(tmpFileName.c_str());

      #else // USE_CURL
              ERR_MSG("Error: to use this functionality you must compile SMIL with the Curl option");
      #endif // USE_CURL
              return img;
          }
          
          ImageFileInfo fInfo;
          if (getFileInfo(filename, fInfo)!=RES_OK)
          {
              ERR_MSG("Can't open file");
              return NULL;
          }
          
          ImageFileHandler<void> *fHandler  = getHandlerForFile<void>(filename);
          ASSERT(fHandler, NULL);
            
          if (fInfo.colorType==ImageFileInfo::COLOR_TYPE_GRAY)
          {
              if (fInfo.scalarType==ImageFileInfo::SCALAR_TYPE_UINT8)
              {
                  Image<UINT8> *img = new Image<UINT8>();
                  auto_ptr< ImageFileHandler<UINT8> > fHandler(getHandlerForFile<UINT8>(filename));
                  if (fHandler->read(filename, *img)==RES_OK)
                    return img;
                  else ERR_MSG("Error reading unsigned 8 bit image");
              }
              else if (fInfo.scalarType==ImageFileInfo::SCALAR_TYPE_UINT16)
              {
                  Image<UINT16> *img = new Image<UINT16>();
                  auto_ptr< ImageFileHandler<UINT16> > fHandler(getHandlerForFile<UINT16>(filename));
                  if (fHandler->read(filename, *img)==RES_OK)
                    return img;
                  else ERR_MSG("Error reading unsigned 16 bit image");
              }
              else if (fInfo.scalarType==ImageFileInfo::SCALAR_TYPE_INT16)
              {
                  Image<INT16> *img = new Image<INT16>();
                  auto_ptr< ImageFileHandler<INT16> > fHandler(getHandlerForFile<INT16>(filename));
                  if (fHandler->read(filename, *img)==RES_OK)
                    return img;
                  else ERR_MSG("Error reading signed 16 bit image");
              }
              else ERR_MSG("Unsupported GRAY data type");
          }
#ifdef SMIL_WRAP_RGB
          else if (fInfo.colorType==ImageFileInfo::COLOR_TYPE_RGB)
          {
              Image<RGB> *img = new Image<RGB>();
              auto_ptr< ImageFileHandler<RGB> > fHandler(getHandlerForFile<RGB>(filename));
              if (fHandler->read(filename, *img)==RES_OK)
                return img;
              else ERR_MSG("Error reading RGB image");
          }
#endif // SMIL_WRAP_RGB
          else
          {
              ERR_MSG("Image data type not supported");
          }
          
        return NULL;
          
    }
    
    
     
} // namespace smil

