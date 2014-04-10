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


#ifdef USE_JPEG

#include "DImageIO.hpp"
#include "DImageIO_JPG.hpp"
#include "DColor.h"

#include <jpeglib.h>

namespace smil
{
    struct JPGHeader
    {
	JPGHeader(const string rw)
	{
	}
	~JPGHeader()
	{
	}
	
    };
    
    RES_T readJPGHeader(FILE *fp, JPGHeader &hStruct)
    {
	return RES_OK;
    }

    RES_T writeJPGHeader(FILE *fp, JPGHeader &hStruct)
    {
	return RES_OK;
    }
    
  
    RES_T getJPGFileInfo(const char* filename, ImageFileInfo &fInfo)
    {
	/* open image file */
	FILE *fp = fopen (filename, "rb");
	
	if (!fp)
	{
	    cout << "Cannot open file " << filename << endl;
	    return RES_ERR_IO;
	}
	
	struct jpeg_error_mgr err_mgr;
	struct jpeg_decompress_struct cinfo;
	
	/* initialize the JPEG decompression object. */
	jpeg_create_decompress(&cinfo);
	cinfo.err = jpeg_std_error(&err_mgr);
	/* specify data source (eg, a file) */
	jpeg_stdio_src(&cinfo, fp);
	/* read file parameters */
	(void) jpeg_read_header(&cinfo, TRUE);
    
	fclose(fp);
	
	fInfo.width = cinfo.image_width;
	fInfo.height = cinfo.image_height;
	fInfo.channels = cinfo.num_components;
	
	switch(cinfo.data_precision)
	{
	  case 8:
	    fInfo.scalarType = ImageFileInfo::SCALAR_TYPE_UINT8; break;
	  case 16:
	    fInfo.scalarType = ImageFileInfo::SCALAR_TYPE_UINT16; break;
	}
	
	switch(cinfo.out_color_space)
	{
	  case JCS_RGB:
	    fInfo.colorType = ImageFileInfo::COLOR_TYPE_RGB; break;
	  default:
	    fInfo.colorType = ImageFileInfo::COLOR_TYPE_UNKNOWN; 
	}
	
	/* destroy cinfo */
	jpeg_destroy_decompress(&cinfo);
    
	return RES_OK;
    }
	
    template <>
    RES_T JPGImageFileHandler<RGB>::read(const char *filename, Image<RGB> &image)
    {
	/* open image file */
	FILE *fp = fopen (filename, "rb");
	
	if (!fp)
	{
	    cout << "Cannot open file " << filename << endl;
	    return RES_ERR_IO;
	}
	
	FileCloser fc(fp);
	
	struct jpeg_error_mgr err_mgr;
	struct jpeg_decompress_struct cinfo;
	
	/* initialize the JPEG decompression object. */
	jpeg_create_decompress(&cinfo);
	cinfo.err = jpeg_std_error(&err_mgr);
	/* specify data source (eg, a file) */
	jpeg_stdio_src(&cinfo, fp);
	/* read file parameters */
	(void) jpeg_read_header(&cinfo, TRUE);
    
	
	size_t width = cinfo.image_width, height = cinfo.image_height;
	
	ASSERT((image.setSize(width, height)==RES_OK), RES_ERR_BAD_ALLOCATION);
	
	ASSERT(cinfo.data_precision==8 && cinfo.num_components==3, "Not a 24bit RGB image", RES_ERR);
	
	JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, width * cinfo.num_components, 1);

	Image<RGB>::sliceType lines = image.getLines();
	MultichannelArray<UINT8,3>::lineType *arrays;
	
	(void) jpeg_start_decompress(&cinfo);
	
	for (size_t j=0;j<height;j++)
	{
	    arrays = lines[j].arrays;
	    jpeg_read_scanlines(&cinfo, buffer, 1);
	    for (size_t i=0;i<width;i++)
	      for (UINT n=0;n<3;n++)
		arrays[n][i] = buffer[0][3*i+n];
	}
	
	(void) jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	
	image.modified();
	
	return RES_OK;
    }


    
    template <>
    RES_T JPGImageFileHandler<RGB>::write(const Image<RGB> &image, const char *filename)
    {
	/* open image file */
	FILE *fp = fopen (filename, "wb");
	
	if (!fp)
	{
	    cout << "Cannot open file " << filename << endl;
	    return RES_ERR_IO;
	}
	
	FileCloser fc(fp);

	struct jpeg_error_mgr err_mgr;
	struct jpeg_compress_struct cinfo;
	
	/* initialize the JPEG compression object. */
	jpeg_create_compress(&cinfo);
	cinfo.err = jpeg_std_error(&err_mgr);
	/* specify data dest (eg, a file) */
	jpeg_stdio_dest(&cinfo, fp);
	
	size_t width = image.getWidth(), height = image.getHeight();
	
        cinfo.image_width = width;      /* image width and height, in pixels */
        cinfo.image_height = height;
        cinfo.input_components = 3;     /* # of color components per pixel */
        cinfo.in_color_space = JCS_RGB; /* colorspace of input image */
        jpeg_set_defaults(&cinfo);
	
	UINT8 *buffer = new UINT8[width*3];

	Image<RGB>::sliceType lines = image.getLines();
	MultichannelArray<UINT8,3>::lineType *arrays;
	
	(void) jpeg_start_compress(&cinfo, TRUE);
	
	for (size_t j=0;j<height;j++)
	{
	    arrays = lines[j].arrays;
	    for (size_t i=0;i<width;i++)
	      for (UINT n=0;n<3;n++)
		buffer[3*i+n] = arrays[n][i];
	    jpeg_write_scanlines(&cinfo, &buffer, 1);
	}
	
	(void) jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
	delete[] buffer;
	
	return RES_OK;
    }
    

} // namespace smil

#endif // USE_JPEG
