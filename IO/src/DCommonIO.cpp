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

#include "IO/include/DCommonIO.h"
#include "Core/include/DErrors.h"

#include <string>
#include <algorithm>
#include <cctype>

#ifdef USE_CURL
#include <curl/curl.h>
#endif // USE_CURL

namespace smil
{
  void ImageFileInfo::printSelf(std::ostream &os)
  {
    std::string s = "";

    os << "Image info in flie " << filename << std::endl;
    os << " filename    " << filename << std::endl;
    os << " channels    " << channels << std::endl;
    switch (colorType) {
      case COLOR_TYPE_GRAY:
        s = "GRAY";
        break;
      case COLOR_TYPE_RGB:
        s = "RGB";
        break;
      case COLOR_TYPE_GA:
        s = "GA";
        break;
      case COLOR_TYPE_RGBA:
        s = "RGBA";
        break;
      case COLOR_TYPE_BINARY:
        s = "BINARY";
        break;
      case COLOR_TYPE_UNKNOWN:
        s = "UNKNOWN";
        break;
      default:
        s = "UNKNOWN";
        break;
    }
    os << " colorType   " << s << std::endl;
    switch(scalarType) {
      case SCALAR_TYPE_UINT8:
        s = "UINT8";
        break;
      case SCALAR_TYPE_UINT16:
        s = "UINT16";
        break;
      case SCALAR_TYPE_INT8:
        s = "INT8";
        break;
      case SCALAR_TYPE_INT16:
        s = "INT16";
        break;
      case SCALAR_TYPE_FLOAT:
        s = "FLOAT";
        break;
      case SCALAR_TYPE_DOUBLE:
        s = "DOUBLE";
        break;
      case SCALAR_TYPE_UNKNOWN:
        s = "UNKNOWN";
        break;
      default:
        s = "UNKNOWN";
        break;
    }
    os << " scalarType  " << s << std::endl;
    switch (fileType) {
      case FILE_TYPE_ASCII:
        s = "ASCII";
        break;
      case FILE_TYPE_BINARY:
        s = "BINARY";
        break;
      default:
        s = "UNKNOWN";
        break;
    }
    os << " fileType    " << s << std::endl;
    os << " size        " << width << "x" << height << "x" << depth << std::endl;
    os << " width       " << width << std::endl;
    os << " height      " << height << std::endl;
    os << " depth       " << depth << std::endl;
  }

  std::string getFileExtension(const char *fileName)
  {
    std::string fName(fileName);
    std::string::size_type idx = fName.rfind('.');
    std::string fExt           = fName.substr(idx + 1);
    transform(fExt.begin(), fExt.end(), fExt.begin(), ::toupper);
    return fExt;
  }

#ifdef USE_CURL
  size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream)
  {
    size_t written;
    written = fwrite(ptr, size, nmemb, stream);
    return written;
  }

  RES_T getHttpFile(const char *url, const char *outfilename)
  {
    CURL *curl_handle;
    FILE *fp;
    CURLcode res;
    curl_handle = curl_easy_init();
    if (curl_handle) {
      SMIL_OPEN(fp, outfilename, "wb");
      curl_easy_setopt(curl_handle, CURLOPT_URL, url);
      curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_data);
      curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, fp);
      curl_easy_setopt(curl_handle, CURLOPT_FAILONERROR, 1);
      curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 3L);
      curl_easy_setopt(curl_handle, CURLOPT_SSL_VERIFYPEER, 0L);
      res = curl_easy_perform(curl_handle);
      if (res != CURLE_OK) {
        std::cout << "Curl error : " << res << std::endl;
      }
      curl_easy_cleanup(curl_handle);
      fclose(fp);
    } else
      res = CURLE_FAILED_INIT;

    ASSERT((res == CURLE_OK), RES_ERR_IO);

    return RES_OK;
  }

  static std::string memoryCallbackBuffer;

  static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb,
                                    void * /*stream*/)
  {
    memoryCallbackBuffer.append((char *) contents, size * nmemb);
    return size * nmemb;
  }

  /**
   * Download file data into a string buffer.
   */
  std::string getHttpFile(const char *url)
  {
    CURL *curl_handle;
    memoryCallbackBuffer.clear();

    //         curl_global_init(CURL_GLOBAL_ALL);
    CURLcode res = CURLE_OK;
    curl_handle  = curl_easy_init();
    if (curl_handle) {
      curl_easy_setopt(curl_handle, CURLOPT_URL, url);
      curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
      curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 3L);
      res = curl_easy_perform(curl_handle);
      curl_easy_cleanup(curl_handle);
    }
    if (res != CURLE_OK) {
      ERR_MSG("curl_easy_perform() failed");
    }
    //         else res = CURLE_FAILED_INIT;

    return memoryCallbackBuffer;
  }
#endif // USE_CURL

} // namespace smil
