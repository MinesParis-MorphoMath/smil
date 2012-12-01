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

#include "DImageIO_PNG.hpp"
#include "DCore.h"

#include <png.h>

template <>
_DIO RES_T readPNG<UINT8>(const char *filename, Image<UINT8> &image)
{

    png_byte magic[8];
    png_structp png_ptr;
    png_infop info_ptr;
    int bit_depth, color_type;
    FILE *fp = NULL;
    png_bytep *row_pointers = NULL;
    png_uint_32 width, height;

    /* open image file */
    fp = fopen (filename, "rb");
    
    if (!fp)
    {
        fprintf (stderr, "error: couldn't open \"%s\"!\n", filename);
        return RES_ERR_IO;
    }

    /* read magic number */
    if (fread (magic, 1, sizeof (magic), fp)!=sizeof(magic) ||
    /* check for valid magic number */
      !png_check_sig (magic, sizeof (magic)))
    {
        fprintf (stderr, "error: \"%s\" is not a valid PNG image!\n",
                 filename);
        fclose (fp);
        return RES_ERR_IO;
    }

    /* create a png read struct */
    png_ptr = png_create_read_struct
              (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
    {
        fclose (fp);
        return RES_ERR_IO;
    }

    /* create a png info struct */
    info_ptr = png_create_info_struct (png_ptr);
    if (!info_ptr)
    {
        fclose (fp);
        png_destroy_read_struct (&png_ptr, NULL, NULL);
        return RES_ERR_IO;
    }


    /* initialize the setjmp for returning properly after a libpng
       error occured */
    if (setjmp (png_jmpbuf (png_ptr)))
    {
        fclose (fp);
        png_destroy_read_struct (&png_ptr, &info_ptr, NULL);

        return RES_ERR_IO;
    }

    /* setup libpng for using standard C fread() function
       with our FILE pointer */
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

    image.setSize(width, height);
//   image->allocate();

    /* setup a pointer array.  Each one points at the begening of a row. */
    row_pointers = image.getLines();

    /* read pixel data using row pointers */
    png_read_image (png_ptr, row_pointers);

    /* finish decompression and release memory */
    png_read_end (png_ptr, NULL);
    png_destroy_read_struct (&png_ptr, &info_ptr, NULL);


    fclose (fp);
    
    image.modified();
    
    return RES_OK;
}




/* write a png file */
template <>
_DIO RES_T writePNG(Image<UINT8> &image, const char *filename)
{
    png_byte color_type = PNG_COLOR_TYPE_GRAY;
    png_byte bit_depth = 8;

    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep * row_pointers = image.getLines();

    /* create file */
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        return RES_ERR_IO;
    }

    /* initialize stuff */
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
    {
        cout << "[write_png_file] png_create_write_struct failed" << endl;
        return RES_ERR_IO;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        cout << "[write_png_file] png_create_info_struct failed" << endl;
        return RES_ERR_IO;
    }

    if (setjmp(png_jmpbuf(png_ptr)))
    {
        cout << "[write_png_file] Error during init_io" << endl;
        return RES_ERR_IO;
    }

    png_init_io(png_ptr, fp);


    /* write header */
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        cout << "[write_png_file] Error during writing header" << endl;
        return RES_ERR_IO;
    }

    png_set_IHDR(png_ptr, info_ptr, image.getWidth(), image.getHeight(),
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        cout << "[write_png_file] Error during writing bytes" << endl;
        return RES_ERR_IO;
    }

    png_write_image(png_ptr, row_pointers);


    /* end write */
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        cout << "[write_png_file] Error during end of write" << endl;
        return RES_ERR_IO;
    }

    png_write_end(png_ptr, NULL);


    fclose(fp);
    return RES_OK;
}


#endif // USE_PNG
