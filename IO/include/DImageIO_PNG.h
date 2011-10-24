#ifndef _D_IMAGE_IO_PNG_H
#define _D_IMAGE_IO_PNG_H

#ifdef USE_PNG

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "DTypes.hpp"
#include "DImage.h"

using namespace std;


_SMIL int readPNG(const char* filename, Image<UINT8> *image);
_SMIL int writePNG(Image<UINT8> *image, const char *filename);


#endif // USE_PNG

#endif // _D_IMAGE_IO_PNG_H
