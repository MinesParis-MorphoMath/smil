#ifndef _D_IMAGE_IO_H
#define _D_IMAGE_IO_H


// #include "D_Types.h"
#include "DImage.h"
#include "DImageIO_BMP.h"

#ifdef USE_PNG
#include "DImageIO_PNG.h"
#endif // USE_PNG


using namespace std;


int read(const char* filename, Image<UINT8> *image);
int write(Image<UINT8> *image, const char *filename);

template <>
Image<UINT8>& Image<UINT8>::operator << (const char *filename);

template <>
Image<UINT8>& Image<UINT8>::operator >> (const char *filename);



#endif // _D_IMAGE_IO_H
