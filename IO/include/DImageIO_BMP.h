#ifndef _D_IMAGE_IO_BMP_H
#define _D_IMAGE_IO_BMP_H

#include "DImageIO.h"
#include <fstream>
#include <iostream> 

#ifndef DWORD
#define DWORD unsigned long
#define WORD unsigned short
#define LONG long
#endif

#define BITMAP_ID 0x4D42        // the universal bitmap ID

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


_SMIL RES_T readBMP(const char* filename, Image<UINT8> *image);
_SMIL RES_T writeBMP(Image<UINT8> *image, const char *filename);

#endif // _D_IMAGE_IO_BMP_H
