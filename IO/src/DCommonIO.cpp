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


#include "DCommonIO.h"
#include <DErrors.h>

#include <string>
#include <algorithm>

#ifdef USE_CURL
#include <curl/curl.h>
#endif // USE_CURL

namespace smil
{


    _DIO string getFileExtension(const char *fileName)
    {
	string fName(fileName);
	string::size_type idx = fName.rfind('.');
	string fExt = fName.substr(idx+1).c_str();
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


    struct MemoryStruct 
    {
	char *memory;
	size_t size;
    };
    
    
    static size_t
    WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
    {
	size_t realsize = size * nmemb;
	struct MemoryStruct *mem = (struct MemoryStruct *)userp;

	mem->memory = (char*)realloc(mem->memory, mem->size + realsize + 1);
	if (mem->memory == NULL) {
	  /* out of memory! */ 
	  printf("not enough memory (realloc returned NULL)\n");
	  exit(EXIT_FAILURE);
	}

	memcpy(&(mem->memory[mem->size]), contents, realsize);
	mem->size += realsize;
	mem->memory[mem->size] = 0;

	return realsize;
    }
    
    

    RES_T getHttpFile(const char *url, const char *outfilename) 
    {
	CURL *curl_handle;
	FILE *fp;
	CURLcode res;
	curl_handle = curl_easy_init();
	if (curl_handle) 
	{
	    fp = fopen(outfilename,"wb");
	    curl_easy_setopt(curl_handle, CURLOPT_URL, url);
	    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_data);
	    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, fp);
	    curl_easy_setopt(curl_handle, CURLOPT_FAILONERROR, 1);
	    res = curl_easy_perform(curl_handle);
	    curl_easy_cleanup(curl_handle);
	    fclose(fp);
	}
	else res = CURLE_FAILED_INIT;
	
	ASSERT((res==CURLE_OK), RES_ERR_IO);

	return RES_OK;
    }

    /**
    * Download file data into a chunk of memory.
    * (Don't forget to free the chunk when done)
    */
    int getHttpFile(const char *url, struct MemoryStruct &chunk) 
    {
	CURL *curl_handle;

	if(chunk.memory)
	  free(chunk.memory);
	chunk.memory = (char*)malloc(1);
	chunk.size = 0;

	curl_global_init(CURL_GLOBAL_ALL);
	CURLcode res;
	curl_handle = curl_easy_init();
	if (curl_handle)
	{
	    curl_easy_setopt(curl_handle, CURLOPT_URL, "http://www.example.com/");
	    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
	    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
	    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
	    res = curl_easy_perform(curl_handle);
	    curl_easy_cleanup(curl_handle);
	    curl_global_cleanup();
	}
	else res = CURLE_FAILED_INIT;

	return res==CURLE_OK;
    }
#endif // USE_CURL
    
} // namespace smil

