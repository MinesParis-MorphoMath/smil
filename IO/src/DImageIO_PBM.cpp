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

#include "Core/include/DErrors.h"
#include "IO/include/private/DImageIO_PBM.hpp"
#include "IO/include/private/DImageIO.hpp"
#include "Core/include/private/DImage.hpp"
#include "Core/include/DColor.h"
#include "Base/include/private/DMeasures.hpp"

namespace smil
{

  RES_T readNetPBMFileInfo(std::ifstream &fp, ImageFileInfo &fInfo,
                           unsigned int &maxval)
  {
    std::string buf;

    getline(fp, buf);

    if ((buf[0] != 'P' && buf[0] != 'p')) {
      ERR_MSG("Error reading NetPBM header");
      return RES_ERR_IO;
    }

    int pbmFileType;
    pbmFileType = atoi(buf.data() + 1);

    switch (pbmFileType) {
      case 1: // Portable BitMap ASCII
        fInfo.colorType = ImageFileInfo::COLOR_TYPE_BINARY;
        fInfo.fileType  = ImageFileInfo::FILE_TYPE_ASCII;
        fInfo.channels  = 1;
        break;
      case 2: // Portable GrayMap ASCII
        fInfo.colorType = ImageFileInfo::COLOR_TYPE_GRAY;
        fInfo.fileType  = ImageFileInfo::FILE_TYPE_ASCII;
        fInfo.channels  = 1;
        break;
      case 3: // Portable PixMap ASCII
        fInfo.colorType = ImageFileInfo::COLOR_TYPE_RGB;
        fInfo.fileType  = ImageFileInfo::FILE_TYPE_ASCII;
        fInfo.channels  = 3;
        break;
      case 4: // Portable BitMap ASCII
        fInfo.colorType = ImageFileInfo::COLOR_TYPE_BINARY;
        fInfo.fileType  = ImageFileInfo::FILE_TYPE_BINARY;
        fInfo.channels  = 1;
        break;
      case 5: // Portable GrayMap ASCII
        fInfo.colorType = ImageFileInfo::COLOR_TYPE_GRAY;
        fInfo.fileType  = ImageFileInfo::FILE_TYPE_BINARY;
        fInfo.channels  = 1;
        break;
      case 6: // Portable PixMap ASCII
        fInfo.colorType = ImageFileInfo::COLOR_TYPE_RGB;
        fInfo.fileType  = ImageFileInfo::FILE_TYPE_BINARY;
        fInfo.channels  = 3;
        break;
      default:
        ERR_MSG("Unknown NetPBM format");
        return RES_ERR_IO;
    }

    std::streampos curpos;
    // Read comments
    do {
      curpos = fp.tellg();
      getline(fp, buf);
    } while (buf[0] == '#');

    // Read image dimensions
    fp.seekg(curpos);
    fp >> fInfo.width >> fInfo.height;

    if (fInfo.colorType != ImageFileInfo::COLOR_TYPE_BINARY) {
      fp >> maxval; // Max pixel value
    }

    fp.seekg(1, std::ios_base::cur); // endl

    fInfo.dataStartPos = fp.tellg();
    fInfo.scalarType   = ImageFileInfo::SCALAR_TYPE_UINT8;

    return RES_OK;
  }

  RES_T readNetPBMFileInfo(const char *filename, ImageFileInfo &fInfo,
                           unsigned int &maxval)
  {
    /* open image file */
    std::ifstream fp(filename, std::ios_base::binary);

    if (!fp.is_open()) {
      std::cout << "Cannot open file " << filename << std::endl;
      return RES_ERR_IO;
    }

    RES_T ret = readNetPBMFileInfo(fp, fInfo, maxval);

    fp.close();
    return ret;
  }

  RES_T PGMImageFileHandler<UINT8>::read(const char   *filename,
                                         Image<UINT8> &image)
  {
    /* open image file */
    std::ifstream fp(filename, std::ios_base::binary);

    if (!fp.is_open()) {
      std::cout << "Cannot open file " << filename << std::endl;
      return RES_ERR_IO;
    }

    ImageFileInfo fInfo;
    unsigned int  maxval;
    ASSERT(readNetPBMFileInfo(fp, fInfo, maxval) == RES_OK, RES_ERR_IO);
    ASSERT(fInfo.colorType == ImageFileInfo::COLOR_TYPE_GRAY,
           "Not an 8bit gray image", RES_ERR_IO);

    int width  = fInfo.width;
    int height = fInfo.height;

    ASSERT((image.setSize(width, height) == RES_OK), RES_ERR_BAD_ALLOCATION);

    if (fInfo.fileType == ImageFileInfo::FILE_TYPE_BINARY) {
      fp.read((char *) image.getPixels(), static_cast<long>(width) * height);
    } else {
      ImDtTypes<UINT8>::lineType pixels = image.getPixels();

      for (size_t i = 0; i < image.getPixelCount(); i++, pixels++) {
        int px;
        fp >> px;
        *((int *) pixels) = px * ImDtTypes<UINT8>::max() / maxval;
      }
    }

    fp.close();

    return RES_OK;
  }

  RES_T PGMImageFileHandler<UINT8>::write(const Image<UINT8> &image,
                                          const char         *filename)
  {
    /* open image file */
    std::ofstream fp(filename, std::ios_base::binary);

    if (!fp.is_open()) {
      std::cout << "Cannot open file " << filename << std::endl;
      return RES_ERR_IO;
    }

    size_t width = image.getWidth(), height = image.getHeight();

    fp << "P5" << std::endl;
    fp << "# " << filename << std::endl;
    fp << width << " " << height << std::endl;
    fp << "255" << std::endl;

    fp.write((char *) image.getPixels(), static_cast<long>(width) * height);

    fp.close();

    return RES_OK;
  }

#ifdef SMIL_WRAP_RGB
  RES_T PGMImageFileHandler<RGB>::read(const char * /*filename*/,
                                       Image<RGB> & /*image*/)
  {
    //         FILE *fp = fopen( filename, "rb" );
    //
    //         ASSERT(fp!=NULL, string("Cannot open file ") + filename + " for
    //         input", RES_ERR_IO);
    //
    //         FileCloser fc(fp);
    //
    //         PBMHeader hStruct;
    //
    //         ASSERT(readPBMHeader(fp, hStruct)==RES_OK);
    //
    //         bmpFileHeader &fHeader = hStruct.fHeader;
    //         bmpInfoHeader &iHeader = hStruct.iHeader;
    //
    //         ASSERT(iHeader.biBitCount==24, "Not an 32bit RGB image",
    //         RES_ERR_IO);
    //
    //         fseek(fp, fHeader.bfOffBits, SEEK_SET);
    //
    //         int width = iHeader.biWidth;
    //         int height = iHeader.biHeight;
    //
    //         ASSERT((image.setSize(width, height)==RES_OK),
    //         RES_ERR_BAD_ALLOCATION);
    //
    //         Image<RGB>::sliceType lines = image.getLines();
    //         MultichannelArray<UINT8,3>::lineType *arrays;
    //         UINT8 *data = new UINT8[width*3];
    //
    //         for (int j=height-1;j>=0;j--)
    //         {
    //             arrays = lines[j].arrays;
    //             ASSERT((fread(data, width*3, 1, fp)!=0), RES_ERR_IO);
    //             for (int i=0;i<width;i++)
    //               for (UINT n=0;n<3;n++)
    //                 arrays[n][i] = data[3*i+(2-n)];
    //         }
    //
    //         delete[] data;
    //
    //         image.modified();
    //
    return RES_OK;
  }
#endif // SMIL_WRAP_RGB

#ifdef SMIL_WRAP_RGB
  RES_T PGMImageFileHandler<RGB>::write(const Image<RGB> & /*image*/,
                                        const char * /*filename*/)
  {
    //         FILE* fp = fopen( filename, "wb" );
    //
    //         if ( fp == NULL )
    //         {
    //             cout << "Error: Cannot open file " << filename << " for
    //             output." << endl; return RES_ERR;
    //         }
    //         bmpFileHeader fHeader;
    //         bmpInfoHeader iHeader;
    //
    //         size_t width = image.getWidth();
    //         size_t height = image.getHeight();
    //
    //         fHeader.bfType = 0x4D42;
    //         fHeader.bfSize = (UINT32)(width*height*3*sizeof(UINT8)) +
    //         sizeof(bmpFileHeader) + sizeof(bmpInfoHeader);
    //         fHeader.bfReserved1 = 0;
    //         fHeader.bfReserved2 = 0;
    //         fHeader.bfOffBits = sizeof(bmpFileHeader) +
    //         sizeof(bmpInfoHeader);
    //
    //         iHeader.biSize = sizeof(bmpInfoHeader);  // number of bytes
    //         required by the struct iHeader.biWidth = (UINT32)width;  // width
    //         in pixels iHeader.biHeight = (UINT32)height;  // height in pixels
    //         iHeader.biPlanes = 1; // number of color planes, must be 1
    //         iHeader.biBitCount = 24; // number of bit per pixel
    //         iHeader.biCompression = 0;// type of compression
    //
    //
    //         //write the bitmap file header
    //         fwrite(&fHeader, sizeof(bmpFileHeader), 1 ,fp);
    //
    //         //write the bitmap image header
    //         fwrite(&iHeader, sizeof(bmpInfoHeader), 1 ,fp);
    //
    //         Image<RGB>::sliceType lines = image.getLines();
    //         MultichannelArray<UINT8,3>::lineType *arrays;
    //         UINT8 *data = new UINT8[width*3];
    //
    //         for (int j=height-1;j>=0;j--)
    //         {
    //             arrays = lines[j].arrays;
    //             for (size_t i=0;i<width;i++)
    //               for (UINT n=0;n<3;n++)
    //                 data[3*i+(2-n)] = arrays[n][i];
    //             ASSERT((fwrite(data, width*3, 1, fp)!=0), RES_ERR_IO);
    //         }
    //
    //         delete[] data;
    //
    // //         Image<UINT8>::lineType *lines = image.getLines();
    // //
    // //         for (size_t i=height-1;i>=0;i--)
    // //             fwrite(lines[i], width*sizeof(UINT8), 1, fp);
    //
    //         fclose(fp);

    return RES_OK;
  }
#endif // SMIL_WRAP_RGB

} // namespace smil
