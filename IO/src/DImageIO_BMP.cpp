/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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


#include "Core/include/DErrors.h"
#include "IO/include/private/DImageIO_BMP.hpp"
#include "IO/include/private/DImageIO.hpp"
#include "Core/include/private/DImage.hpp"
#include "Core/include/DColor.h"

namespace smil
{
    #ifndef DWORD
    #define DWORD unsigned long
    #define WORD unsigned short
    #define LONG long
    #endif

    #define BITMAP_ID 0x4D42        // the universal bitmap ID

#ifndef BI_RGB
    enum {
      BI_RGB,        // An uncompressed format.
      BI_RLE8,        // A run-length encoded (RLE) format for bitmaps with 8 bpp. The compression format is a 2-byte format consisting of a count byte followed by a byte containing a color index.
      BI_RLE4,        // An RLE format for bitmaps with 4 bpp. The compression format is a 2-byte format consisting of a count byte followed by two word-length color indexes.
      BI_BITFIELDS,        // Specifies that the bitmap is not compressed and that the color table consists of three DWORD color masks that specify the red, green, and blue components, respectively, of each pixel. This is valid when used with 16- and 32-bpp bitmaps.
      BI_JPEG,        // Windows 98/Me, Windows 2000/XP: Indicates that the image is a JPEG image.
      BI_PNG, };
#endif 
      
    #pragma pack(push, 1) // force the compiler to pack the structs
    struct bmpFileHeader
    {
        UINT16 bfType;  // file type
        UINT32 bfSize;  // size in bytes of the bitmap file
        UINT16 bfReserved1;  // reserved; must be 0
        UINT16 bfReserved2;  // reserved; must be 0
        UINT32 bfOffBits;  // offset in bytes from the bitmapfileheader to the bitmap bits
    };

    struct bmpInfoHeader
    {
        UINT32 biSize;  // number of bytes required by the struct
        UINT32 biWidth;  // width in pixels
        UINT32 biHeight;  // height in pixels
        UINT16 biPlanes; // number of color planes, must be 1
        UINT16 biBitCount; // number of bit per pixel
        UINT32 biCompression;// type of compression
        UINT32 biSizeImage;  //size of image in bytes
        UINT32 biXPelsPerMeter;  // number of pixels per meter in x axis
        UINT32 biYPelsPerMeter;  // number of pixels per meter in y axis
        UINT32 biClrUsed;  // number of colors used by the bitmap
        UINT32 biClrImportant;  // number of colors that are important
    };

    #pragma pack(pop)

    struct BMPHeader
    {
        BMPHeader()
        {
        }
        ~BMPHeader()
        {
        }
        
        bmpFileHeader fHeader;
        bmpInfoHeader iHeader;
    };
    
    
    RES_T readBMPHeader(FILE *fp, BMPHeader &hStruct)
    {
        bmpFileHeader &fHeader = hStruct.fHeader;
        bmpInfoHeader &iHeader = hStruct.iHeader;
        
        //read the bitmap file header
        ASSERT(fread(&fHeader, sizeof(bmpFileHeader), 1 ,fp));

        //verify that this is a bmp file by check bitmap id
        ASSERT((fHeader.bfType == 0x4D42));
        
        //read the bitmap info header
        ASSERT(fread(&iHeader, sizeof(bmpInfoHeader), 1, fp));
        
        ASSERT(iHeader.biCompression==BI_RGB, "Compressed BMP files are not (yet) supported", RES_ERR_IO);
        
        return RES_OK;
    }

    RES_T getBMPFileInfo(const char* filename, ImageFileInfo &fInfo)
    {
        /* open image file */
        FILE *fp = fopen (filename, "rb");
        
        if (!fp)
        {
            cout << "Cannot open file " << filename << endl;
            return RES_ERR_IO;
        }
        
        BMPHeader hStruct;
        ASSERT(readBMPHeader(fp, hStruct)==RES_OK);
        
        fclose(fp);
        
        bmpInfoHeader &iHeader = hStruct.iHeader;
        
        fInfo.width = iHeader.biWidth;
        fInfo.height = iHeader.biHeight;
        fInfo.scalarType = ImageFileInfo::SCALAR_TYPE_UINT8;
        fInfo.channels = iHeader.biBitCount / 8;
        
        switch(iHeader.biBitCount)
        {
          case 8:
            fInfo.colorType = ImageFileInfo::COLOR_TYPE_GRAY; break;
          case 24:
            fInfo.colorType = ImageFileInfo::COLOR_TYPE_RGB; break;
          default:
            fInfo.colorType = ImageFileInfo::COLOR_TYPE_UNKNOWN; 
        }
        
        return RES_OK;
    }
    
    template <>
    RES_T BMPImageFileHandler<UINT8>::read(const char *filename, Image<UINT8> &image)
    {
        FILE *fp = fopen( filename, "rb" );

        ASSERT(fp!=NULL, string("Cannot open file ") + filename + " for input", RES_ERR_IO);
        
        FileCloser fc(fp);
        
        BMPHeader hStruct;
        
        ASSERT(readBMPHeader(fp, hStruct)==RES_OK);
        
        bmpFileHeader &fHeader = hStruct.fHeader;
        bmpInfoHeader &iHeader = hStruct.iHeader;
        
        ASSERT(iHeader.biBitCount==8, "Not an 8bit gray image", RES_ERR_IO);
        
        UINT nColors = iHeader.biClrUsed;
        UINT8 r,g,b,a, *lut = new UINT8[nColors];

        // read the color table
          
        for (UINT j=0; j<nColors; j++) {
            ASSERT(fread(&b, 1, 1, fp));
            ASSERT(fread(&g, 1, 1, fp));
            ASSERT(fread(&r, 1, 1, fp));
            ASSERT(fread(&a, 1, 1, fp));
            lut[j] = (UINT8)(((UINT32)r+(UINT32)g+(UINT32)b)/3) ;

        }
    
        // at this point it is safer to  
        // move file pointer to the beginning of bitmap data (skip palette information)
        fseek(fp, fHeader.bfOffBits, SEEK_SET);

        int width = iHeader.biWidth;
        int height = iHeader.biHeight;

        ASSERT((image.setSize(width, height)==RES_OK), RES_ERR_BAD_ALLOCATION);
        
        Image<UINT8>::sliceType lines = image.getLines();
        Image<UINT8>::lineType curLine;

        // The scan line must be word-aligned. This means in multiples of 4.
        int scanlineSize = (width%4==0) ? width : (width-width%4)+4;
        UINT8 *scanBuf = new UINT8[scanlineSize];
        
        for (int j=height-1;j>=0;j--)
        {
            curLine = lines[j];
            ASSERT((fread(scanBuf, scanlineSize, 1, fp)!=0), RES_ERR_IO);
            for(int i=0; i< width; i++) 
                curLine[i] = lut[scanBuf[i]];
        }
        delete[] lut;
        delete[] scanBuf;
        image.modified();

        return RES_OK;
    }

#ifdef SMIL_WRAP_RGB      
    template <>
    RES_T BMPImageFileHandler<RGB>::read(const char *filename, Image<RGB> &image)
    {
        FILE *fp = fopen( filename, "rb" );

        ASSERT(fp!=NULL, string("Cannot open file ") + filename + " for input", RES_ERR_IO);
        
        FileCloser fc(fp);
        
        BMPHeader hStruct;
        
        ASSERT(readBMPHeader(fp, hStruct)==RES_OK);
        
        bmpFileHeader &fHeader = hStruct.fHeader;
        bmpInfoHeader &iHeader = hStruct.iHeader;
        
        ASSERT(iHeader.biBitCount==24, "Not an 32bit RGB image", RES_ERR_IO);
        
        fseek(fp, fHeader.bfOffBits, SEEK_SET);

        int width = iHeader.biWidth;
        int height = iHeader.biHeight;

        ASSERT((image.setSize(width, height)==RES_OK), RES_ERR_BAD_ALLOCATION);
        
        Image<RGB>::sliceType lines = image.getLines();
        MultichannelArray<UINT8,3>::lineType *arrays;
        UINT8 *data = new UINT8[width*3];

        for (int j=height-1;j>=0;j--)
        {
            arrays = lines[j].arrays;
            ASSERT((fread(data, width*3, 1, fp)!=0), RES_ERR_IO);
            for (int i=0;i<width;i++)
              for (UINT n=0;n<3;n++)
                arrays[n][i] = data[3*i+(2-n)];
        }
        
        delete[] data;
        
        image.modified();

        return RES_OK;
    }
#endif // SMIL_WRAP_RGB  

    template <>
    RES_T BMPImageFileHandler<UINT8>::write(const Image<UINT8> &image, const char *filename)
    {
        FILE* fp = fopen( filename, "wb" );

        if ( fp == NULL )
        {
            cout << "Error: Cannot open file " << filename << " for output." << endl;
            return RES_ERR;
        }
        bmpFileHeader fHeader;
        bmpInfoHeader iHeader;

        size_t width = image.getWidth();
        size_t height = image.getHeight();

        int nColors = 256;
        
        int scanlineSize = (width%4==0) ? width : (width-width%4)+4;
        
        fHeader.bfType = 0x4D42;
        fHeader.bfReserved1 = 0;
        fHeader.bfReserved2 = 0;
        fHeader.bfOffBits = sizeof(bmpFileHeader) + sizeof(bmpInfoHeader) + nColors*4;
        fHeader.bfSize = fHeader.bfOffBits + UINT32(scanlineSize*height);

        iHeader.biSize = sizeof(bmpInfoHeader);  // number of bytes required by the struct
        iHeader.biWidth = (UINT32)width;  // width in pixels
        iHeader.biHeight = (UINT32)height;  // height in pixels
        iHeader.biPlanes = 1; // number of color planes, must be 1
        iHeader.biBitCount = 8; // number of bit per pixel
        iHeader.biCompression = 0;// type of compression
        iHeader.biSizeImage = 0;  //size of image in bytes
        iHeader.biClrUsed = nColors;  // number of colors used by the bitmap
        iHeader.biClrImportant = nColors;  // number of colors that are important


        //write the bitmap file header
        fwrite(&fHeader, sizeof(bmpFileHeader), 1 ,fp);

        //write the bitmap image header
        fwrite(&iHeader, sizeof(bmpInfoHeader), 1 ,fp);

        // write palette
        for (int i=0;i<256;i++)
        {
            fputc(i, fp);
            fputc(i, fp);
            fputc(i, fp);
            fputc(0, fp);
        }

        Image<UINT8>::lineType *lines = image.getLines();

        // The scan line must be word-aligned. This means in multiples of 4.
        int scanlinePadSize = (width%4==0) ? 0 : 4-width%4;
        UINT *scanlinePad = new UINT[scanlinePadSize];

        for (int i=height-1;i>=0;i--)
        {
            fwrite(lines[i], width, 1, fp);
            if (scanlinePadSize)
              fwrite(scanlinePad, scanlinePadSize, 1, fp);
        }

        fclose(fp);
        delete[] scanlinePad;

        return RES_OK;
    }

#ifdef SMIL_WRAP_RGB  
    template <>
    RES_T BMPImageFileHandler<RGB>::write(const Image<RGB> &image, const char *filename)
    {
        FILE* fp = fopen( filename, "wb" );

        if ( fp == NULL )
        {
            cout << "Error: Cannot open file " << filename << " for output." << endl;
            return RES_ERR;
        }
        bmpFileHeader fHeader;
        bmpInfoHeader iHeader;

        size_t width = image.getWidth();
        size_t height = image.getHeight();

        fHeader.bfType = 0x4D42;
        fHeader.bfSize = (UINT32)(width*height*3*sizeof(UINT8)) + sizeof(bmpFileHeader) + sizeof(bmpInfoHeader);
        fHeader.bfReserved1 = 0;
        fHeader.bfReserved2 = 0;
        fHeader.bfOffBits = sizeof(bmpFileHeader) + sizeof(bmpInfoHeader);

        iHeader.biSize = sizeof(bmpInfoHeader);  // number of bytes required by the struct
        iHeader.biWidth = (UINT32)width;  // width in pixels
        iHeader.biHeight = (UINT32)height;  // height in pixels
        iHeader.biPlanes = 1; // number of color planes, must be 1
        iHeader.biBitCount = 24; // number of bit per pixel
        iHeader.biCompression = 0;// type of compression


        //write the bitmap file header
        fwrite(&fHeader, sizeof(bmpFileHeader), 1 ,fp);

        //write the bitmap image header
        fwrite(&iHeader, sizeof(bmpInfoHeader), 1 ,fp);

        Image<RGB>::sliceType lines = image.getLines();
        MultichannelArray<UINT8,3>::lineType *arrays;
        UINT8 *data = new UINT8[width*3];

        for (int j=height-1;j>=0;j--)
        {
            arrays = lines[j].arrays;
            for (size_t i=0;i<width;i++)
              for (UINT n=0;n<3;n++)
                data[3*i+(2-n)] = arrays[n][i];
            ASSERT((fwrite(data, width*3, 1, fp)!=0), RES_ERR_IO);
        }
        
        delete[] data;
        
//         Image<UINT8>::lineType *lines = image.getLines();
// 
//         for (size_t i=height-1;i>=0;i--)
//             fwrite(lines[i], width*sizeof(UINT8), 1, fp);

        fclose(fp);

        return RES_OK;
    }
#endif // SMIL_WRAP_RGB  

} // namespace smil
