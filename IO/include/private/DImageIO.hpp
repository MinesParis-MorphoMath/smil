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


#include <string>
#include <algorithm>



#include "DImageIO_BMP.hpp"
#include "DImageIO_RAW.hpp"
#include "DImageIO_VTK.hpp"

#ifdef USE_PNG
#include "DImageIO_PNG.hpp"
#endif // USE_PNG

namespace smil
{
  
    /** 
    * \addtogroup IO
    */
    /*@{*/
    
    extern _DIO string getFileExtension(const char *fileName);

    #ifdef USE_CURL
    extern _DIO RES_T getHttpFile(const char *url, const char *outfilename);
    #endif // USE_CURL

    class FileCloser
    {
    public:
	FileCloser(FILE *_fp)
	{
	    fp = _fp;
	}
	~FileCloser()
	{
	    if (fp)
	      fclose(fp);
	}
    protected:
	FILE *fp;
    };

    /**
    * Read image file
    */
    template <class T>
    RES_T read(const char* filename, Image<T> &image)
    {
	string fileExt = getFileExtension(filename);
	string filePrefix = (string(filename).substr(0, 7));
	string tmpFileName;

	RES_T res;

	if (filePrefix=="http://")
	{
    #ifdef USE_CURL
	    tmpFileName = "_smilTmpIO." + fileExt;
	    if (getHttpFile(filename, tmpFileName.c_str())!=RES_OK)
	    {
		ERR_MSG(string("Error downloading file ") + filename);
		return RES_ERR;
	    }
	    res = read(tmpFileName.c_str(), image);
	    remove(tmpFileName.c_str());

    #else // USE_CURL
	    ERR_MSG("Error: to use this functionality you must compile smil with the Curl option");
	    res = RES_ERR;
    #endif // USE_CURL
	    return res;
	}

	if (fileExt=="BMP")
	    res = readBMP(filename, image);

    #ifdef USE_PNG
	else if (fileExt=="PNG")
	    res = readPNG(filename, image);
    #endif // USE_PNG

	else if (fileExt=="VTK")
	    res = readVTK(filename, image);

	else
	{
	    cout << "File type not supported" << endl;
	    res = RES_ERR;
	}

	return res;
    }

    /**
    * Read a stack of 2D images
    * 
    * The output 3D image will have the width and height of the first (2D) image and the number of images for depth.
    */
    template <class T>
    RES_T read(const vector<string> fileList, Image<T> &image)
    {
	UINT nFiles = fileList.size();
	if (nFiles==0)
	  return RES_ERR;
	
	vector<string>::const_iterator it = fileList.begin();
	
	Image<T> tmpIm;
	ASSERT((read((*it++).c_str(), tmpIm)==RES_OK));
	
	size_t w = tmpIm.getWidth(), h = tmpIm.getHeight();
	ImageFreezer freezer(image);
	
	ASSERT((image.setSize(w, h, nFiles)==RES_OK));
	ASSERT((fill(image, T(0))==RES_OK));
	ASSERT((copy(tmpIm, 0, 0, 0, image, 0, 0, 0)==RES_OK));
	
	size_t z = 1;
	
	while(it!=fileList.end())
	{
	    ASSERT((read((*it).c_str(), tmpIm)==RES_OK));
	    ASSERT((copy(tmpIm, 0, 0, 0, image, 0, 0, z)==RES_OK));
	    it++;
	    z++;
	}
	
	return RES_OK;
    }

    /**
    * Write image file
    */
    template <class T>
    RES_T write(Image<T> &image, const char *filename)
    {
	string fileExt = getFileExtension(filename);
	RES_T res;

	if (fileExt=="BMP")
	    res = writeBMP(image, filename);

    #ifdef USE_PNG
	else if (fileExt=="PNG")
	    res = writePNG(image, filename);
    #endif // USE_PNG

	else if (fileExt=="RAW")
	    res = writeRAW(image, filename);

	else if (fileExt=="VTK")
	    res = writeVTK(image, filename);

	else
	{
	    cout << "File type not supported" << endl;
	    res = RES_ERR;
	}

	return res;
    }

/*@}*/

} // namespace smil



#endif // _D_IMAGE_IO_HPP
