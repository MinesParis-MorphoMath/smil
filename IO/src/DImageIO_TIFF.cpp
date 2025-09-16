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


#ifdef USE_TIFF

#include "IO/include/private/DImageIO.hpp"
#include "IO/include/private/DImageIO_TIFF.hpp"
#include "Core/include/DColor.h"

#include <tiffio.h>

namespace smil
{
    struct TIFFHeader
    {
      TIFFHeader(const std::string /*rw*/)
        {
        }
        ~TIFFHeader()
        {
        }
        
    };
    
    RES_T readTIFFHeader(FILE * /*fp*/, TIFFHeader &/*hStruct*/)
    {
        return RES_OK;
    }

    RES_T writeTIFFHeader(FILE * /*fp*/, TIFFHeader &/*hStruct*/)
    {
        return RES_OK;
    }
    
  
    RES_T getTIFFFileInfo(const char* filename, ImageFileInfo &fInfo)
    {
        /* open image file */
        TIFF *tif=TIFFOpen(filename, "r");
        
        if (!tif)
        {
          std::cout << "Cannot open file " << filename << std::endl;
            return RES_ERR_IO;
        }
        
        uint32_t w, h;
        uint16_t nbits, nsamples;
        
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &nbits);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);

        TIFFClose(tif);
        
        fInfo.width = w;
        fInfo.height = h;
        fInfo.channels = nsamples;
        
        switch(nbits)
        {
          case 8:
            fInfo.scalarType = ImageFileInfo::SCALAR_TYPE_UINT8; break;
          case 16:
            fInfo.scalarType = ImageFileInfo::SCALAR_TYPE_UINT16; break;
        }
        
        switch(nsamples)
        {
          case 1:
            fInfo.colorType = ImageFileInfo::COLOR_TYPE_GRAY; break;
          case 3:
            fInfo.colorType = ImageFileInfo::COLOR_TYPE_RGB; break;
          case 4:
            fInfo.colorType = ImageFileInfo::COLOR_TYPE_RGBA; break;
          default:
            fInfo.colorType = ImageFileInfo::COLOR_TYPE_UNKNOWN; 
        }
        
        return RES_OK;
    }
        
    template <class T>
    RES_T StandardTIFFRead(const char *filename, Image<T> &image)
    {
        /* open image file */
        TIFF *tif=TIFFOpen(filename, "r");
        
        if (!tif)
        {
          std::cout << "Cannot open file " << filename << std::endl;
            return RES_ERR_IO;
        }
        
        uint32_t width, height;
        uint16_t nbits, nsamples;
        
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &nbits);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
        
        ASSERT(nbits==8*sizeof(T) && nsamples==1, "Bad image type", RES_ERR);
        ASSERT((image.setSize(width, height)==RES_OK), RES_ERR_BAD_ALLOCATION);
        
        typename ImDtTypes<T>::sliceType lines = image.getLines();
        
        for (size_t j=0;j<height;j++)
          TIFFReadScanline(tif, lines[j], j);
        
        image.modified();
        
        TIFFClose(tif);
        
        return RES_OK;
    }
    
    RES_T TIFFImageFileHandler<UINT8>::read(const char *filename, Image<UINT8> &image)
    {
        return StandardTIFFRead(filename, image);
    }

    RES_T TIFFImageFileHandler<UINT16>::read(const char *filename, Image<UINT16> &image)
    {
        return StandardTIFFRead(filename, image);
    }

    RES_T TIFFImageFileHandler<RGB>::read(const char *filename, Image<RGB> &image)
    {
        /* open image file */
        TIFF *tif=TIFFOpen(filename, "r");
        
        if (!tif)
        {
          std::cout << "Cannot open file " << filename << std::endl;
            return RES_ERR_IO;
        }
        
        uint32_t width, height;
        uint16_t nbits, nsamples;
        
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &nbits);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
        
        ASSERT(nbits==8 && nsamples==3, "Not a 24bit RGB image", RES_ERR);
        ASSERT((image.setSize(width, height)==RES_OK), RES_ERR_BAD_ALLOCATION);
        
        Image<RGB>::sliceType lines = image.getLines();
        MultichannelArray<UINT8,3>::lineType *arrays;
        
        UINT8* raster = (UINT8*) _TIFFmalloc(width * 3 *sizeof (UINT8));
        
        for (size_t j=0;j<height;j++)
        {
            arrays = lines[j].arrays;
            TIFFReadScanline(tif, raster, j);
            for (size_t i=0;i<width;i++)
              for (UINT n=0;n<3;n++)
                arrays[n][i] = raster[3*i+n];
        }
        
        _TIFFfree(raster);
        TIFFClose(tif);
        
        image.modified();
        
        return RES_OK;
    }

    template <class T>
    RES_T StandardTIFFWrite(const Image<T> &image, const char *filename)
    {
        /* open image file */
        TIFF *tif=TIFFOpen(filename, "w");
        
        if (!tif)
        {
          std::cout << "Cannot open file " << filename << std::endl;
            return RES_ERR_IO;
        }
        
        uint32_t width = image.getWidth(), height = image.getHeight();
        uint16_t nbits = 8*sizeof(T), nsamples = 1;
        
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, nbits);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, nsamples);
        
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISWHITE);
        
        typename ImDtTypes<T>::sliceType lines = image.getLines();
        
        for (size_t j=0;j<height;j++)
          TIFFWriteScanline(tif, lines[j], j);
        
        TIFFClose(tif);
        
        return RES_OK;
    }
    
    RES_T TIFFImageFileHandler<UINT8>::write(const Image<UINT8> &image, const char *filename)
    {
        return StandardTIFFWrite(image, filename);
    }
    
    RES_T TIFFImageFileHandler<UINT16>::write(const Image<UINT16> &image, const char *filename)
    {
        return StandardTIFFWrite(image, filename);
    }
    
    RES_T TIFFImageFileHandler<RGB>::write(const Image<RGB> &image, const char *filename)
    {
        /* open image file */
        TIFF *tif=TIFFOpen(filename, "w");
        
        if (!tif)
        {
          std::cout << "Cannot open file " << filename << std::endl;
            return RES_ERR_IO;
        }
        
        uint32_t width = image.getWidth(), height = image.getHeight();
        
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 3);
        
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        
        Image<RGB>::sliceType lines = image.getLines();
        MultichannelArray<UINT8,3>::lineType *arrays;
        
        UINT8* raster = (UINT8*) _TIFFmalloc(width * 3 *sizeof (UINT8));
        
        for (size_t j=0;j<height;j++)
        {
            arrays = lines[j].arrays;
            for (size_t i=0;i<width;i++)
              for (UINT n=0;n<3;n++)
                raster[3*i+n] = arrays[n][i];
            TIFFWriteScanline(tif, raster, j);
        }
        
        _TIFFfree(raster);
        TIFFClose(tif);
        
        return RES_OK;
    }

} // namespace smil

#endif // USE_TIFF
