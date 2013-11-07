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


#ifdef USE_PNG

#include "DImageIO.hpp"
#include "DImageIO_PNG.hpp"

#include <png.h>

namespace smil
{

    struct PNGFileInfo : public ImageFileInfo
    {
	PNGFileInfo(const string rw)
	{
	    if (rw=="r")
	    {
	      readMode = true;
	      /* create a png read struct */
	      png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	    }
	    else
	    {
	      readMode = false;
	      /* create a png write struct */
	      png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	    }
	    
	    /* create a png info struct */
	    info_ptr = png_create_info_struct (png_ptr);
	}
	~PNGFileInfo()
	{
	    if (readMode)
	      png_destroy_read_struct (&png_ptr, &info_ptr, NULL);
	    else
	      png_destroy_write_struct (&png_ptr, &info_ptr);
	}
	
	png_structp png_ptr;
	png_infop info_ptr;
	
	bool readMode;
    };
    
  
    RES_T readPNGHeader(const char *filename, PNGFileInfo &fInfo)
    {
	png_byte magic[8];
	png_structp &png_ptr = fInfo.png_ptr;
	png_infop &info_ptr = fInfo.info_ptr;
	int bit_depth, color_type;
	FILE *&fp = fInfo.fileHandle;
	png_uint_32 width, height;

	/* open image file */
	fp = fopen (filename, "rb");
	
	if (!fp)
	{
	    cout << "Cannot open file " << filename << endl;
	    return RES_ERR_IO;
	}
	
	/* read magic number */
	/* check for valid magic number */
	ASSERT( (fread (magic, 1, sizeof (magic), fp)==sizeof(magic) ||	  png_check_sig (magic, sizeof (magic))), string(filename) + " is not a valid PNG image!", RES_ERR_IO );


	/* initialize the setjmp for returning properly after a libpng
	  error occured */
	if (setjmp (png_jmpbuf (png_ptr)))
	    return RES_ERR_IO;

	/* setup libpng for using standard C fread() function with our FILE pointer */
	png_init_io (png_ptr, fp);

	/* tell libpng that we have already read the magic number */
	png_set_sig_bytes (png_ptr, sizeof (magic));

	/* read png info */
	png_read_info (png_ptr, info_ptr);

	/* get some usefull information from header */
	bit_depth = png_get_bit_depth (png_ptr, info_ptr);
	color_type = png_get_color_type (png_ptr, info_ptr);

	/* convert index color images to RGB images */
	if (color_type == PNG_COLOR_TYPE_PALETTE)
	    png_set_palette_to_rgb (png_ptr);

	/* convert 1-2-4 bits grayscale images to 8 bits
	  grayscale. */
	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
	    png_set_expand_gray_1_2_4_to_8 (png_ptr);

	if (png_get_valid (png_ptr, info_ptr, PNG_INFO_tRNS))
	    png_set_tRNS_to_alpha (png_ptr);

	if (bit_depth == 16)
	    png_set_strip_16 (png_ptr);
	else if (bit_depth < 8)
	    png_set_packing (png_ptr);

	/* update info structure to apply transformations */
	png_read_update_info (png_ptr, info_ptr);

	/* retrieve updated information */
	png_get_IHDR (png_ptr, info_ptr,
		      (png_uint_32*)(&width),
		      (png_uint_32*)(&height),
		      &bit_depth, &color_type,
		      NULL, NULL, NULL);

	fInfo.width = width;
	fInfo.height = height;
	fInfo.bit_depth = bit_depth;
	fInfo.channels = info_ptr->channels;
	
	switch(color_type)
	{
	  case PNG_COLOR_TYPE_GRAY:
	    fInfo.color_type = ImageFileInfo::COLOR_TYPE_GRAY; break;
	  case PNG_COLOR_TYPE_RGB:
	    fInfo.color_type = ImageFileInfo::COLOR_TYPE_RGB; break;
	  case PNG_COLOR_TYPE_RGB_ALPHA:
	    fInfo.color_type = ImageFileInfo::COLOR_TYPE_RGBA; break;
	  case PNG_COLOR_TYPE_GRAY_ALPHA:
	    fInfo.color_type = ImageFileInfo::COLOR_TYPE_GA; break;
	  default:
	    fInfo.color_type = ImageFileInfo::COLOR_TYPE_UNKNOWN; 
	}
	
	return RES_OK;
    }

    RES_T writePNGHeader(const char *filename, PNGFileInfo &fInfo)
    {
	png_byte color_type;
	
	switch(fInfo.color_type)
	{
	  case ImageFileInfo::COLOR_TYPE_GRAY:
	    color_type = PNG_COLOR_TYPE_GRAY; break;
	  case ImageFileInfo::COLOR_TYPE_RGB:
	    color_type = PNG_COLOR_TYPE_RGB; break;
	  case ImageFileInfo::COLOR_TYPE_RGBA:
	    color_type = PNG_COLOR_TYPE_RGB_ALPHA; break;
	  case ImageFileInfo::COLOR_TYPE_GA:
	    color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
	}
	
	png_structp &png_ptr = fInfo.png_ptr;
	png_infop &info_ptr = fInfo.info_ptr;

	/* create file */
	FILE *&fp = fInfo.fileHandle;
	fp = fopen(filename, "wb");
	if (!fp)
	    return RES_ERR_IO;

	png_init_io(png_ptr, fp);

	/* write header */
	png_set_IHDR(png_ptr, info_ptr, fInfo.width, fInfo.height,
		    fInfo.bit_depth, color_type, PNG_INTERLACE_NONE,
		    PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	return RES_OK;
    }

    template <>
    RES_T readPNG<UINT8>(const char *filename, Image<UINT8> &image)
    {
	PNGFileInfo fInfo("r");
	
	ASSERT(readPNGHeader(filename, fInfo)==RES_OK);
	
	ASSERT(fInfo.bit_depth==8 && fInfo.channels==1, "Not a 8bit gray image", RES_ERR);
	
	ASSERT((image.setSize(fInfo.width, fInfo.height)==RES_OK), RES_ERR_BAD_ALLOCATION);

	/* setup a pointer array.  Each one points at the begening of a row. */
	png_bytep *row_pointers = image.getLines();

	/* read pixel data using row pointers */
	png_read_image (fInfo.png_ptr, row_pointers);
	
	/* finish decompression and release memory */
	png_read_end (fInfo.png_ptr, NULL);

	image.modified();
    }

    template <>
    RES_T readPNG<RGB>(const char *filename, Image<RGB> &image)
    {
	PNGFileInfo fInfo("r");
	
	readPNGHeader(filename, fInfo);
	size_t width = fInfo.width, height = fInfo.height;
	
	ASSERT((image.setSize(width, height)==RES_OK), RES_ERR_BAD_ALLOCATION);
	
	ASSERT(fInfo.bit_depth==8 && fInfo.channels==3, "Not a 24bit RGB image", RES_ERR);
	
	typedef UINT8* datap;
	datap *data = new datap[height];
	for (size_t j=0;j<height;j++)
	  data[j] = new UINT8[width*3];

	/* setup a pointer array.  Each one points at the begening of a row. */
	png_bytep *row_pointers = data;

	/* read pixel data using row pointers */
	png_read_image (fInfo.png_ptr, row_pointers);
	/* finish decompression and release memory */
	png_read_end (fInfo.png_ptr, NULL);

	Image<RGB>::sliceType lines = image.getLines();
	MultichannelArray<UINT8,3>::lineType *arrays;
	datap curline;
	
	for (size_t j=0;j<height;j++)
	{
	    arrays = lines[j].arrays;
	    curline = data[j];
	    for (size_t i=0;i<width;i++)
	      for (UINT n=0;n<3;n++)
		arrays[n][i] = curline[3*i+n];
	    delete[] data[j];
	}
	delete[] data;
	
	image.modified();
    }


    template <>
    RES_T writePNG(Image<UINT8> &image, const char *filename)
    {
	PNGFileInfo fInfo("w");
	fInfo.width = image.getWidth();
	fInfo.height = image.getHeight();
	fInfo.bit_depth = 8;
	fInfo.color_type = ImageFileInfo::COLOR_TYPE_GRAY;
	fInfo.channels = 1;
	
	ASSERT(writePNGHeader(filename, fInfo)==RES_OK);
	
	png_bytep * row_pointers = image.getLines();
	png_write_image(fInfo.png_ptr, row_pointers);
	png_write_end(fInfo.png_ptr, NULL);

	return RES_OK;
    }
    
    template <>
    RES_T writePNG(Image<RGB> &image, const char *filename)
    {
	PNGFileInfo fInfo("w");
	fInfo.width = image.getWidth();
	fInfo.height = image.getHeight();
	fInfo.bit_depth = 8;
	fInfo.color_type = ImageFileInfo::COLOR_TYPE_RGB;
	fInfo.channels = 1;
	
	ASSERT(writePNGHeader(filename, fInfo)==RES_OK);
	
	size_t width = fInfo.width, height = fInfo.height;
	
	typedef UINT8* datap;
	datap *data = new datap[height];
	for (size_t j=0;j<height;j++)
	  data[j] = new UINT8[width*3];


	Image<RGB>::sliceType lines = image.getLines();
	MultichannelArray<UINT8,3>::lineType *arrays;
	datap curline;
	
	for (size_t j=0;j<height;j++)
	{
	    arrays = lines[j].arrays;
	    curline = data[j];
	    for (size_t i=0;i<width;i++)
	      for (UINT n=0;n<3;n++)
		curline[3*i+n] = arrays[n][i];
	}
	
	png_bytep *row_pointers = data;
	png_write_image(fInfo.png_ptr, row_pointers);
	png_write_end(fInfo.png_ptr, NULL);

	for (size_t j=0;j<height;j++)
	    delete[] data[j];
	delete[] data;

	return RES_OK;
    }
    
    BaseImage *createFromPNG(const char* filename)
    {
	PNGFileInfo fInfo("r");
	
	ASSERT(readPNGHeader(filename, fInfo)==RES_OK, NULL);
	
	if (fInfo.color_type==ImageFileInfo::COLOR_TYPE_GRAY)
	{
	    Image<UINT8> *img = new Image<UINT8>();
	    readPNG(filename, *img);
	    return img;
	}
	else if (fInfo.color_type==ImageFileInfo::COLOR_TYPE_RGB)
	{
	    Image<RGB> *img = new Image<RGB>();
	    readPNG(filename, *img);
	    return img;
	}
	else
	{
	    ERR_MSG("File type not supported");
	    return NULL;
	}
    }

} // namespace smil

#endif // USE_PNG
