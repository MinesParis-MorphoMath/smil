#ifndef _D_IMAGE_IO_PNG_H
#define _D_IMAGE_IO_PNG_H

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "D_Types.h"
#include "DImage.h"

using namespace std;


int readPNGFile (const char* filename, Image<UINT8> *image);

int writePNGFile (Image<UINT8> *image, const char *filename);

template <>
Image<UINT8>& Image<UINT8>::operator << (const char *filename)
{
    readPNGFile(filename, this);
    modified();
    return *this;
}

template <>
Image<UINT8>& Image<UINT8>::operator >> (const char *filename)
{
    writePNGFile(this, filename);
    return *this;
}
#endif // _D_IMAGE_IO_PNG_H
